#!/usr/bin/env python3
"""
SAE学習スクリプト
L1正則化とk-sparseの両方に対応
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from sae_model import SAE


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


def load_chunks(chunk_dir: Path):
    """chunkファイルを読み込んで結合"""
    chunks = []
    chunk_files = sorted(chunk_dir.glob("chunk_*.pt"))
    
    if len(chunk_files) == 0:
        raise ValueError(f"No chunk files found in {chunk_dir}")
    
    print(f"Loading {len(chunk_files)} chunks...")
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunk = torch.load(chunk_file)
        chunks.append(chunk)
    
    data = torch.cat(chunks, dim=0)
    print(f"Total data shape: {data.shape}")
    return data


def standardize_data(data: torch.Tensor):
    """データを標準化"""
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True) + 1e-5
    data_norm = (data - mean) / std
    return data_norm, mean, std


def train_sae(
    sae: SAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    log_every: int,
    device: torch.device,
    log_file: Optional[Path] = None,
):
    """SAEを学習"""
    sae.train()
    base_sae = sae.module if isinstance(sae, nn.DataParallel) else sae
    step = 0
    log_entries = []
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            
            loss, recon_loss, reg_loss = base_sae.loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % log_every == 0:
                global_sparsity, _ = base_sae.compute_sparsity(base_sae.encoder(batch).relu())
                
                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "reg_loss": reg_loss.item(),
                    "global_sparsity": global_sparsity,
                }
                log_entries.append(log_entry)
                
                print(
                    f"epoch {epoch} step {step}: "
                    f"loss={loss.item():.4e}, "
                    f"recon={recon_loss.item():.4e}, "
                    f"reg={reg_loss.item():.4e}, "
                    f"sparsity={global_sparsity:.4f}"
                )
                
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
            
            step += 1
    
    return log_entries


def main():
    parser = argparse.ArgumentParser(description="Train SAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--layer", type=int, help="Override layer from config")
    parser.add_argument("--signal", type=str, choices=["state", "delta"], help="Override signal from config")
    
    args = parser.parse_args()
    
    # 設定ファイルを読み込み
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で上書き
    if args.layer:
        config["layers"] = [args.layer]
    if args.signal:
        config["signal"] = args.signal
    
    model_name = config["model_name"]
    model_dir_name = model_name.split("/")[-1].replace("-", "_")
    layers = config["layers"]
    signal = config["signal"]
    sae_config = config["sae"]
    train_config = config["train"]
    devices = config.get("devices", [0])

    output_root = resolve_output_root(config)
    model_dir = output_root / model_dir_name
    copy_config(args.config, model_dir)
    
    device = torch.device(f"cuda:{devices[0]}")
    
    # 各層についてSAEを学習
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Training SAE for layer {layer}, signal={signal}")
        print(f"{'='*60}")
        
        # データをロード
        chunk_dir = model_dir / f"layer_{layer}" / signal
        if not chunk_dir.exists():
            print(f"Warning: {chunk_dir} does not exist. Skipping layer {layer}.")
            continue
        data = load_chunks(chunk_dir)
        
        # 標準化
        data_norm, mean, std = standardize_data(data)
        input_dim = data_norm.shape[1]
        
        # SAEの設定
        hidden_factor = sae_config.get("hidden_factor", 8)
        hidden_dim = input_dim * hidden_factor
        mode = sae_config.get("mode", "l1")
        l1_lambda = sae_config.get("l1_lambda")
        k_frac = sae_config.get("k_frac", 0.1)
        
        # SAEを作成
        sae = SAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            mode=mode,
            l1_lambda=l1_lambda,
            k_frac=k_frac,
        ).to(device)
        
        # DataParallelで並列化
        if len(devices) > 1:
            sae = nn.DataParallel(sae, device_ids=devices)
        
        # オプティマイザー
        lr = float(train_config.get("lr", 1e-3))
        optimizer = torch.optim.Adam(
            sae.module.parameters() if len(devices) > 1 else sae.parameters(),
            lr=lr,
        )
        
        # データローダー
        dataset = TensorDataset(data_norm)
        train_loader = DataLoader(
            dataset,
            batch_size=train_config.get("batch_size", 1024),
            shuffle=True,
            num_workers=train_config.get("num_workers", 4),
        )
        
        # ログファイル
        output_dir = model_dir / f"layer_{layer}" / signal
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "train_log.jsonl"
        
        # 学習
        num_epochs = train_config.get("num_epochs", 5)
        log_every = train_config.get("log_every", 100)
        
        train_sae(
            sae=sae,
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            log_every=log_every,
            device=device,
            log_file=log_file,
        )
        
        # チェックポイントを保存
        checkpoint = {
            "state_dict": sae.module.state_dict() if len(devices) > 1 else sae.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "mode": mode,
            "l1_lambda": l1_lambda,
            "k_frac": k_frac,
            "mean": mean.cpu(),
            "std": std.cpu(),
            "layer": layer,
            "model_name": model_name,
            "signal": signal,
            "hidden_factor": hidden_factor,
        }
        
        checkpoint_path = output_dir / "sae_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
