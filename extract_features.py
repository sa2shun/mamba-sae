#!/usr/bin/env python3
"""
特徴抽出スクリプト
Mamba/Transformerモデルから中間state h_tまたは更新量Δh_tを抽出し、.ptファイルとして保存
"""

import argparse
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM


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


def batch_iter(dataset, batch_size: int, max_docs: Optional[int] = None):
    """データセットからバッチを生成するイテレータ"""
    batch = []
    used = 0
    for i, ex in enumerate(dataset):
        if max_docs is not None and used >= max_docs:
            break
        text = ex["text"].strip()
        if not text:
            continue
        batch.append(text)
        used += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model_and_tokenizer(model_name: str, model_type: str, device_ids: List[int], dtype: torch.dtype):
    """モデルとトークナイザーをロード"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "dtype": dtype,            # torch_dtype -> dtype で警告回避
        "use_safetensors": True,   # safetensors を優先
    }

    if model_type == "pythia":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **common_kwargs,
        )
    elif model_type == "mamba":
        model = MambaForCausalLM.from_pretrained(
            model_name,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    device = torch.device(f"cuda:{device_ids[0]}")
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    return model, tokenizer, device


def _get_layer_module(model, model_type: str, layer_index: int):
    """Return the underlying layer module for residual capture (transformer only)."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    if model_type != "transformer":
        raise ValueError("Residual signals are only implemented for transformer backbones.")
    if hasattr(base, "gpt_neox"):
        layers = base.gpt_neox.layers
    elif hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
    else:
        raise ValueError("Unsupported transformer architecture for residual capture.")
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} out of range (max {len(layers)-1}).")
    return layers[layer_index]


@contextmanager
def capture_residual_stream(model, model_type: str, layer_index: int, signal: str):
    """Context manager to capture resid_pre_mlp / resid_post_mlp activations."""
    storage: List[torch.Tensor] = []
    handles = []

    layer = _get_layer_module(model, model_type, layer_index)

    if signal == "resid_pre_mlp":
        def hook(module, inputs, output):
            storage.append(inputs[0].detach().float().cpu())

        handles.append(layer.mlp.register_forward_hook(hook))
    elif signal == "resid_post_mlp":
        def hook(module, inputs, output):
            storage.append(output.detach().float().cpu())

        handles.append(layer.register_forward_hook(hook))
    else:
        raise ValueError(f"Unknown residual signal: {signal}")

    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


def extract_features(
    model,
    tokenizer,
    device,
    dataset,
    target_layer: int,
    signal: str,  # "state" | "delta" | "resid_pre_mlp" | "resid_post_mlp"
    max_length: int,
    batch_size: int,
    max_docs: Optional[int],
    max_tokens_total: Optional[int],
    output_dir: Path,
    chunk_size: int = 100000,
    model_type: str = "transformer",
):
    """特徴を抽出してchunkごとに保存"""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    total_tokens = 0
    chunk_idx = 0

    # For residual signals we need the 0-based layer index
    layer_module_index = target_layer - 1  # hidden_states uses +1 indexing
    num_devices = len(model.device_ids) if isinstance(model, nn.DataParallel) else 1
    
    with torch.no_grad():
        for batch_texts in tqdm(batch_iter(dataset, batch_size=batch_size, max_docs=max_docs), desc="Extracting features"):
            if max_tokens_total is not None and total_tokens >= max_tokens_total:
                break
            
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            if signal in ("state", "delta"):
                out = model(
                    **enc,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hs = out.hidden_states[target_layer]  # [B, T, d]
            else:
                # residual streams (transformer only)
                if num_devices > 1:
                    print("Warning: resid_* signals with DataParallel may be slower; consider single GPU for hooks.")
                with capture_residual_stream(model, model_type, layer_module_index, signal) as storage:
                    _ = model(
                        **enc,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                if len(storage) == 0:
                    continue
                hs = torch.cat(storage, dim=0)  # [B, T, d]

            attn = enc["attention_mask"]  # [B, T]

            B, T, D = hs.shape
            for b in range(B):
                valid_len = int(attn[b].sum().item())
                if valid_len <= 1:
                    continue

                h_seq = hs[b, :valid_len, :].float().cpu()  # [L, d]

                if signal == "state" or signal.startswith("resid_"):
                    features = h_seq  # [L, d]
                elif signal == "delta":
                    if valid_len <= 1:
                        continue
                    features = h_seq[1:] - h_seq[:-1]  # [L-1, d]
                else:
                    raise ValueError(f"Unknown signal: {signal}")

                all_features.append(features)
                total_tokens += features.shape[0]

                # chunk_sizeに達したら保存
                current_size = sum(f.shape[0] for f in all_features)
                if current_size >= chunk_size:
                    chunk = torch.cat(all_features, dim=0)
                    chunk_path = output_dir / f"chunk_{chunk_idx:03d}.pt"
                    torch.save(chunk, chunk_path)
                    print(f"Saved {chunk_path} with shape {chunk.shape}")
                    all_features = []
                    chunk_idx += 1

                if max_tokens_total is not None and total_tokens >= max_tokens_total:
                    break
            
            if max_tokens_total is not None and total_tokens >= max_tokens_total:
                break

    # 残りを保存
    if len(all_features) > 0:
        chunk = torch.cat(all_features, dim=0)
        chunk_path = output_dir / f"chunk_{chunk_idx:03d}.pt"
        torch.save(chunk, chunk_path)
        print(f"Saved {chunk_path} with shape {chunk.shape}")
    
    # メタ情報を保存
    model_config = model.module.config if hasattr(model, "module") else model.config
    model_name_meta = getattr(model_config, "name_or_path", getattr(model_config, "_name_or_path", "unknown"))
    
    meta = {
        "model_name": model_name_meta,
        "target_layer": target_layer,
        "signal": signal,
        "d_model": D,
        "total_samples": total_tokens,
        "num_chunks": chunk_idx + 1,
        "max_length": max_length,
        "batch_size": batch_size,
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Total samples: {total_tokens}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from language models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-name", type=str, help="Override model name from config")
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
    if args.model_name:
        config["model_name"] = args.model_name
    if args.layer:
        config["layers"] = [args.layer]
    if args.signal:
        config["signal"] = args.signal

    model_name = config["model_name"]
    model_type = config.get("model_type", "mamba" if "mamba" in model_name.lower() else "transformer")
    layers = config["layers"]
    signals = config.get("signals")
    if signals is None:
        signals = [config.get("signal", "state")]
    if args.signal:
        signals = [args.signal]

    dataset_config = config["dataset"]
    devices = config.get("devices", [0])
    extract_config = config.get("extract", {})
    max_length = extract_config.get("max_length", 1024)
    batch_size = extract_config.get("batch_size", 32)
    max_docs = dataset_config.get("max_docs")
    max_tokens_total = dataset_config.get("max_tokens_total")
    chunk_size = extract_config.get("chunk_size", 100000)
    
    # データセットをロード
    ds = load_dataset(dataset_config["name"], dataset_config.get("config"), split=dataset_config["split"])
    
    # モデルとトークナイザーをロード
    dtype = torch.float16
    model, tokenizer, device = load_model_and_tokenizer(model_name, model_type, devices, dtype)
    
    # モデル名からディレクトリ名を生成（例: state-spaces/mamba-1.4b-hf -> mamba_1.4b）
    output_root = resolve_output_root(config)
    model_dir_name = model_name.split("/")[-1].replace("-", "_")
    model_dir = output_root / model_dir_name
    copy_config(args.config, model_dir)
    
    # 各層について特徴抽出
    for signal in signals:
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Processing layer {layer}, signal={signal}")
            print(f"{'='*60}")
            
            output_dir = model_dir / f"layer_{layer}" / signal
            extract_features(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=ds,
                target_layer=layer,
                signal=signal,
                max_length=max_length,
                batch_size=batch_size,
                max_docs=max_docs,
                max_tokens_total=max_tokens_total,
                chunk_size=chunk_size,
                output_dir=output_dir,
                model_type=model_type,
            )


if __name__ == "__main__":
    main()
