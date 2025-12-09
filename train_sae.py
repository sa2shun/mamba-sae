#!/usr/bin/env python3
"""
SAE training script with full hyperparameter grid support.

- Supports state / delta (and future signals) across multiple layers
- Hyperparameter grid: hidden_factor, k_frac, l1_lambda, eq_alpha
- Seeds: run replicates for variance estimates
- Cosine schedule with warmup, gradient clipping, AdamW
- Validation split with best-checkpoint selection
- Automatic input normalization (per model/layer/signal) using 1–2M tokens
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sae_model import SAE


class TensorIndexDataset(Dataset):
    """Lightweight view over a tensor with precomputed indices."""

    def __init__(self, data: torch.Tensor, indices: torch.Tensor):
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_root(config: dict) -> Path:
    """Resolve output root, expanding ~ and env vars (e.g., /raid/${USER})."""
    root = config.get("output_root", "runs")
    root_path = Path(os.path.expanduser(os.path.expandvars(root)))
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def copy_config(config_path: str, target_dir: Path):
    """Copy the YAML config into the run directory for reproducibility."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, target_dir / Path(config_path).name)
    except FileNotFoundError:
        print(f"Warning: config file {config_path} not found; skipping copy.")


def list_chunk_files(chunk_dir: Path) -> List[Path]:
    files = sorted(chunk_dir.glob("chunk_*.pt"))
    if len(files) == 0:
        raise ValueError(f"No chunk files found in {chunk_dir}")
    return files


def load_chunks(chunk_dir: Path, max_tokens: Optional[int]) -> torch.Tensor:
    """Load chunks sequentially (CPU) up to max_tokens."""
    chunk_files = list_chunk_files(chunk_dir)
    tensors = []
    total = 0
    for cf in chunk_files:
        x = torch.load(cf, map_location="cpu").float()
        if max_tokens is not None and total + x.shape[0] > max_tokens:
            x = x[: max_tokens - total]
        tensors.append(x)
        total += x.shape[0]
        if max_tokens is not None and total >= max_tokens:
            break
    data = torch.cat(tensors, dim=0)
    print(f"Loaded {len(data)} samples from {len(tensors)} chunks (max_tokens={max_tokens}).")
    return data


def compute_stats_streaming(chunk_files: Sequence[Path], max_tokens: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Streaming mean/std using Welford to avoid memory blowup."""
    mean = None
    m2 = None
    count = 0
    for cf in chunk_files:
        x = torch.load(cf, map_location="cpu").float()
        if max_tokens is not None and count + x.shape[0] > max_tokens:
            x = x[: max_tokens - count]
        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if mean is None:
            mean = batch_mean
            m2 = batch_var * batch_n
            count = batch_n
        else:
            delta = batch_mean - mean
            total = count + batch_n
            mean = mean + delta * batch_n / total
            m2 = m2 + batch_var * batch_n + delta.pow(2) * count * batch_n / total
            count = total

        if max_tokens is not None and count >= max_tokens:
            break

    if mean is None:
        raise ValueError("Failed to compute stats; no data seen.")

    variance = m2 / max(count, 1)
    std = torch.sqrt(variance + 1e-6)
    return mean, std, count


def prepare_data(chunk_dir: Path, dataset_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute (or load) mean/std then load + normalize data tensor."""
    chunk_files = list_chunk_files(chunk_dir)
    stats_path = chunk_dir / "stats.pt"
    stats_tokens = dataset_cfg.get("stats_tokens", dataset_cfg.get("max_tokens_total"))

    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"]
        std = stats["std"]
        stats_seen = int(stats.get("count", 0))
        print(f"Loaded cached stats from {stats_path} (count={stats_seen}).")
    else:
        mean, std, stats_seen = compute_stats_streaming(chunk_files, stats_tokens)
        torch.save({"mean": mean, "std": std, "count": stats_seen}, stats_path)
        print(f"Saved stats to {stats_path} (count={stats_seen}).")

    data = load_chunks(chunk_dir, max_tokens=dataset_cfg.get("max_tokens_total"))
    data_norm = (data - mean) / (std + 1e-6)
    return data_norm, mean, std, stats_seen


def split_indices(n: int, val_split: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_size = max(1, int(n * val_split))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def build_loaders(
    data: torch.Tensor,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    train_idx, val_idx = split_indices(len(data), val_split, seed)
    train_ds = TensorIndexDataset(data, train_idx)
    val_ds = TensorIndexDataset(data, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_idx, val_idx


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_on_loader(
    sae: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 1e-4,
    sample_limit: int = 50000,
):
    sae.eval()
    base = sae.module if isinstance(sae, nn.DataParallel) else sae
    total_recon = 0.0
    total_count = 0
    z_samples = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_hat, z = base(batch)
            recon = F.mse_loss(x_hat, batch, reduction="sum")
            total_recon += recon.item()
            total_count += batch.size(0)
            if len(z_samples) < sample_limit:
                z_samples.append(z.detach().cpu())

    if z_samples:
        z_cat = torch.cat(z_samples, dim=0)
        global_sparsity, per_feature = base.compute_sparsity(z_cat, threshold=threshold)
        dead_rate = (per_feature < 1e-4).float().mean().item()
        mean_firing = per_feature.mean().item()
        median_firing = per_feature.median().item()
        active_mean = base.num_active_per_sample(z_cat, threshold=threshold).float().mean().item()
    else:
        global_sparsity = dead_rate = mean_firing = median_firing = active_mean = float("nan")

    recon_mean = total_recon / max(total_count, 1)
    return {
        "recon_loss": recon_mean,
        "global_sparsity": global_sparsity,
        "dead_rate": dead_rate,
        "mean_firing": mean_firing,
        "median_firing": median_firing,
        "active_mean": active_mean,
        "samples": total_count,
    }


def compute_pca_cumvar(x: torch.Tensor, k: int = 256) -> List[float]:
    """Return cumulative variance ratio for top-k components."""
    k = min(k, x.shape[0] - 1, x.shape[1])
    if k < 2:
        return []
    x_center = x - x.mean(dim=0, keepdim=True)
    _, S, _ = torch.pca_lowrank(x_center, q=k)
    var = S.pow(2)
    cumvar = torch.cumsum(var, dim=0) / var.sum()
    return cumvar.cpu().tolist()


def collect_activation_stats(
    sae: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    sample_size: int = 20000,
    threshold: float = 1e-4,
):
    base = sae.module if isinstance(sae, nn.DataParallel) else sae
    num_samples = min(sample_size, len(data))
    idx = torch.randperm(len(data))[:num_samples]
    subset = data[idx]

    recon_errors = []
    active_counts = []
    z_all = []

    loader = DataLoader(subset, batch_size=1024, shuffle=False)
    base.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_hat, z = base(batch)
            err = F.mse_loss(x_hat, batch, reduction="none").sum(dim=1)
            recon_errors.append(err.cpu())
            active_counts.append(base.num_active_per_sample(z, threshold=threshold).cpu())
            z_all.append(z.cpu())

    recon_errors = torch.cat(recon_errors, dim=0)
    active_counts = torch.cat(active_counts, dim=0)
    z_cat = torch.cat(z_all, dim=0)

    global_sparsity, per_feature = base.compute_sparsity(z_cat, threshold=threshold)
    dead_rate = (per_feature < 1e-4).float().mean().item()

    code_pca = compute_pca_cumvar(z_cat, k=256)
    input_pca = compute_pca_cumvar(subset, k=min(256, subset.shape[1]))

    stats = {
        "global_sparsity": global_sparsity,
        "dead_rate": dead_rate,
        "mean_firing": per_feature.mean().item(),
        "median_firing": per_feature.median().item(),
        "active_mean": active_counts.float().mean().item(),
        "active_median": active_counts.median().item(),
        "active_p95": active_counts.quantile(0.95).item(),
        "code_pca_cumvar": code_pca,
        "input_pca_cumvar": input_pca,
        "sample_size": num_samples,
        "recon_error_mean": recon_errors.mean().item(),
        "recon_error_median": recon_errors.median().item(),
    }

    return stats, {
        "active_counts": active_counts,
        "recon_errors": recon_errors,
        "per_feature_firing": per_feature,
    }


def format_run_name(hidden_factor: int, k_frac: float, l1_lambda: float, eq_alpha: float, seed: int) -> str:
    def fmt(x: float) -> str:
        s = f"{x}".replace("-", "n")
        return s.replace(".", "p")

    return f"hf{hidden_factor}_k{fmt(k_frac)}_l1{fmt(l1_lambda)}_eq{fmt(eq_alpha)}_seed{seed}"


def save_checkpoint(
    path: Path,
    sae: nn.Module,
    meta: Dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = sae.module.state_dict() if isinstance(sae, nn.DataParallel) else sae.state_dict()
    torch.save({**meta, "state_dict": state_dict}, path)


def train_one_run(
    data: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    model_name: str,
    layer: int,
    signal: str,
    devices: List[int],
    train_cfg: dict,
    hyper: dict,
    seed: int,
    base_dir: Path,
    stats_seen: int,
):
    set_seed(seed)
    device = torch.device(f"cuda:{devices[0]}")

    batch_size = int(train_cfg.get("batch_size", 1024))
    val_split = float(train_cfg.get("val_split", 0.1))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    log_every = int(train_cfg.get("log_every", 100))
    num_epochs = int(train_cfg.get("num_epochs", 5))
    warmup_steps = int(train_cfg.get("warmup_steps", 1000))
    clip_grad = float(train_cfg.get("clip_grad_norm", 1.0))

    train_loader, val_loader, train_idx, val_idx = build_loaders(
        data=data,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    input_dim = data.shape[1]
    hidden_dim = input_dim * int(hyper["hidden_factor"])

    sae = SAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k_frac=hyper["k_frac"],
        l1_lambda=hyper["l1_lambda"],
        eq_alpha=hyper["eq_alpha"],
        eq_target=hyper.get("eq_target", hyper["k_frac"]),
    ).to(device)

    if len(devices) > 1:
        sae = nn.DataParallel(sae, device_ids=devices)
    base = sae.module if isinstance(sae, nn.DataParallel) else sae

    params = base.parameters() if len(devices) == 1 else sae.module.parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    total_steps = num_epochs * len(train_loader)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    run_name = format_run_name(
        hidden_factor=int(hyper["hidden_factor"]),
        k_frac=hyper["k_frac"],
        l1_lambda=hyper["l1_lambda"],
        eq_alpha=hyper["eq_alpha"],
        seed=seed,
    )
    run_dir = base_dir / "grid" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    best_val = float("inf")
    best_checkpoint_path = run_dir / "sae_checkpoint.pt"
    step = 0

    for epoch in range(num_epochs):
        sae.train()
        epoch_recon = 0.0
        epoch_count = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)

            total_loss, recon_loss, l1_loss, eq_loss = base.loss(batch)

            optimizer.zero_grad()
            total_loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=clip_grad)
            optimizer.step()
            scheduler.step()

            epoch_recon += recon_loss.item() * batch.size(0)
            epoch_count += batch.size(0)

            if step % log_every == 0:
                global_sparsity, _ = base.compute_sparsity(base.encode(batch))
                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss_total": total_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "eq_loss": eq_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "global_sparsity": global_sparsity,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            step += 1

        train_recon_mean = epoch_recon / max(epoch_count, 1)

        # Validation
        val_metrics = evaluate_on_loader(sae, val_loader, device)
        if val_metrics["recon_loss"] < best_val:
            best_val = val_metrics["recon_loss"]
            save_checkpoint(
                best_checkpoint_path,
                sae,
                meta={
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "k_frac": hyper["k_frac"],
                    "l1_lambda": hyper["l1_lambda"],
                    "eq_alpha": hyper["eq_alpha"],
                    "eq_target": hyper.get("eq_target", hyper["k_frac"]),
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "layer": layer,
                    "model_name": model_name,
                    "signal": signal,
                    "seed": seed,
                    "hidden_factor": int(hyper["hidden_factor"]),
                    "stats_tokens": stats_seen,
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                },
            )

        # Append epoch-level summary for quick monitoring
        epoch_summary = {
            "epoch": epoch,
            "train_recon": train_recon_mean,
            "val_recon": val_metrics["recon_loss"],
            "val_sparsity": val_metrics["global_sparsity"],
        }
        with open(run_dir / "epoch_summary.jsonl", "a") as f:
            f.write(json.dumps(epoch_summary) + "\n")

    # Final detailed stats using best checkpoint weights
    best_sae_state = torch.load(best_checkpoint_path, map_location=device)
    base.load_state_dict(best_sae_state["state_dict"])

    detailed_stats, raw_arrays = collect_activation_stats(sae, data, device)
    torch.save(raw_arrays, run_dir / "sample_metrics.pt")

    summary = {
        "model": model_name,
        "layer": layer,
        "signal": signal,
        "seed": seed,
        "run_name": run_name,
        "hidden_factor": int(hyper["hidden_factor"]),
        "k_frac": hyper["k_frac"],
        "l1_lambda": hyper["l1_lambda"],
        "eq_alpha": hyper["eq_alpha"],
        "eq_target": hyper.get("eq_target", hyper["k_frac"]),
        "best_val_recon": best_val,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "stats_tokens": stats_seen,
        "detailed_stats": detailed_stats,
        "checkpoint": str(best_checkpoint_path),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Append to grid-level results
    grid_log = base_dir / "grid_results.jsonl"
    with open(grid_log, "a") as f:
        f.write(json.dumps(summary) + "\n")

    return summary


def expand_grid(sae_cfg: dict) -> List[dict]:
    """Generate all hyperparameter combinations."""
    def ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    grid = []
    for hf in ensure_list(sae_cfg.get("hidden_factor", [8])):
        for k_frac in ensure_list(sae_cfg.get("k_frac", [0.1])):
            for l1 in ensure_list(sae_cfg.get("l1_lambda", [0.0])):
                for eq in ensure_list(sae_cfg.get("eq_alpha", [0.0])):
                    grid.append(
                        {
                            "hidden_factor": int(hf),
                            "k_frac": float(k_frac),
                            "l1_lambda": float(l1),
                            "eq_alpha": float(eq),
                            "eq_target": sae_cfg.get("eq_target"),
                        }
                    )
    return grid


def main():
    parser = argparse.ArgumentParser(description="Train SAE grid")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--layer", type=int, help="Override layer from config")
    parser.add_argument("--signal", type=str, choices=["state", "delta", "resid_pre_mlp", "resid_post_mlp"], help="Override signal from config")
    parser.add_argument("--limit-grid", type=int, help="Run only the first N hyperparameter combinations (for quick tests)")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Config file {args.config} is empty or invalid.")

    model_name = config["model_name"]
    model_dir_name = model_name.split("/")[-1].replace("-", "_")
    layers = config["layers"] if args.layer is None else [args.layer]
    signals = config.get("signals")
    if signals is None:
        signals = [config.get("signal", "state")]
    if args.signal:
        signals = [args.signal]

    sae_config = config["sae"]
    train_config = config["train"]
    seeds = train_config.get("seeds", [0, 1, 2])
    devices = config.get("devices", [0])

    output_root = resolve_output_root(config)
    model_dir = output_root / model_dir_name
    copy_config(args.config, model_dir)

    grid = expand_grid(sae_config)
    if args.limit_grid:
        grid = grid[: args.limit_grid]
        print(f"Running only the first {len(grid)} grid points due to --limit-grid.")

    dataset_cfg = config["dataset"]

    for signal in signals:
        for layer in layers:
            print(f"\n{'='*70}")
            print(f"Preparing data for layer {layer}, signal={signal}")
            print(f"{'='*70}")

            chunk_dir = model_dir / f"layer_{layer}" / signal
            if not chunk_dir.exists():
                print(f"Warning: {chunk_dir} does not exist. Skipping.")
                continue

            data_norm, mean, std, stats_seen = prepare_data(chunk_dir, dataset_cfg)
            base_dir = chunk_dir

            print(f"Data ready: {data_norm.shape[0]} samples, dim={data_norm.shape[1]}")
            for hyper in grid:
                for seed in seeds:
                    summary = train_one_run(
                        data=data_norm,
                        mean=mean,
                        std=std,
                        model_name=model_name,
                        layer=layer,
                        signal=signal,
                        devices=devices,
                        train_cfg=train_config,
                        hyper=hyper,
                        seed=seed,
                        base_dir=base_dir,
                        stats_seen=stats_seen,
                    )
                    print(
                        f"Finished {summary['run_name']}: val_recon={summary['best_val_recon']:.4e}, "
                        f"sparsity={summary['detailed_stats']['global_sparsity']:.4f}"
                    )


if __name__ == "__main__":
    main()
