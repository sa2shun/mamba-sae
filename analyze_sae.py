#!/usr/bin/env python3
"""
SAE解析スクリプト
- sparsity分布の可視化
- half-life分布の計算（state-SAEの場合）
- 各featureのtop activation contextsの表示
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
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

from sae_model import SAE


def load_model_and_tokenizer(model_name: str, model_type: str, device_ids: List[int], dtype: torch.dtype):
    """モデルとトークナイザーをロード"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "dtype": dtype,            # torch_dtype -> dtype で警告回避
        "use_safetensors": True,   # CVE 回避のため safetensors 優先
    }

    if model_type == "mamba":
        model = MambaForCausalLM.from_pretrained(model_name, **common_kwargs)
    else:  # transformer
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    
    device = torch.device(f"cuda:{device_ids[0]}")
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    
    return model, tokenizer, device


def load_sae_checkpoint(checkpoint_path: Path, device: torch.device):
    """SAEチェックポイントをロード"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    sae = SAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        mode=checkpoint["mode"],
        l1_lambda=checkpoint.get("l1_lambda"),
        k_frac=checkpoint.get("k_frac", 0.1),
    ).to(device)
    
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    
    return sae, checkpoint


def compute_sparsity_distribution(sae: SAE, data: torch.Tensor, threshold: float = 1e-3):
    """スパース性分布を計算"""
    sae.eval()
    with torch.no_grad():
        z_all = []
        batch_size = 1024
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(next(sae.parameters()).device)
            _, z = sae(batch)
            z_all.append(z.cpu())
        
        z_all = torch.cat(z_all, dim=0)
        
        global_sparsity, per_feature_sparsity = sae.compute_sparsity(z_all, threshold=threshold)
        
        return global_sparsity, per_feature_sparsity, z_all


def autocorr_1d(x: np.ndarray, max_lag: int):
    """1次元配列の自己相関を計算"""
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2
    ac = ac[mid:mid+max_lag+1] / var
    return ac


def compute_half_lives(z_long: torch.Tensor, max_lag: int = 64, num_features: int = 256):
    """half-life分布を計算"""
    num_features_total = z_long.shape[1]
    step_feat = max(1, num_features_total // num_features)
    feat_indices = np.arange(0, num_features_total, step_feat)
    
    taus = []
    for i in feat_indices:
        series = z_long[:, i].numpy()
        ac = autocorr_1d(series, max_lag=max_lag)
        # ACFが0.5を初めて下回るラグを探す
        below = np.where(ac < 0.5)[0]
        if len(below) == 0:
            tau = max_lag
        else:
            tau = below[0]
        taus.append(tau)
    
    return np.array(taus)


def extract_top_contexts(
    model,
    tokenizer,
    device,
    sae: SAE,
    checkpoint: dict,
    dataset,
    target_layer: int,
    signal: str,
    max_length: int = 2048,
    num_docs: int = 200,
    top_k: int = 10,
    num_features: int = 10,
):
    """各featureのtop activation contextsを抽出"""
    # 長いテキストを生成
    texts = []
    for i, ex in enumerate(dataset):
        if len(texts) >= num_docs:
            break
        text = ex["text"].strip()
        if text:
            texts.append(text)
    
    long_text = " ".join(texts)
    
    enc_long = tokenizer(
        long_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    
    with torch.no_grad():
        out_long = model(
            **enc_long,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hs_long = out_long.hidden_states[target_layer][0]  # [T, d]
        T, D = hs_long.shape
        
        if signal == "state":
            features_long = hs_long.float()  # [T, d] on model device
        elif signal == "delta":
            features_long = (hs_long[1:].float() - hs_long[:-1].float())  # [T-1, d] on model device
        else:
            raise ValueError(f"Unknown signal: {signal}")
        
        # 標準化
        mean = checkpoint["mean"].to(features_long.device)
        std = checkpoint["std"].to(features_long.device)
        features_long_norm = (features_long - mean) / std
        
        # SAEに通す
        _, z_long = sae(features_long_norm)
        z_long = z_long.cpu()
    
    tokens = tokenizer.convert_ids_to_tokens(enc_long["input_ids"][0].tolist())
    
    # 各featureについてtop-k activation contextsを抽出
    contexts = {}
    for feature_id in range(min(num_features, z_long.shape[1])):
        series = z_long[:, feature_id].numpy()
        top_pos = np.argsort(series)[-top_k:][::-1]
        
        feature_contexts = []
        for rank, pos in enumerate(top_pos, 1):
            left = max(0, pos - 5)
            right = min(len(tokens) - 1, pos + 5)
            span = tokens[left:right+1]
            feature_contexts.append({
                "rank": rank,
                "pos": int(pos),
                "activation": float(series[pos]),
                "context": " ".join(span),
            })
        
        contexts[feature_id] = feature_contexts
    
    return contexts, tokens


def main():
    parser = argparse.ArgumentParser(description="Analyze trained SAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--layer", type=int, help="Override layer from config")
    parser.add_argument("--signal", type=str, choices=["state", "delta"], help="Override signal from config")
    parser.add_argument("--skip-half-life", action="store_true", help="Skip half-life computation")
    
    args = parser.parse_args()
    
    # 設定ファイルを読み込み
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install it with: pip install pyyaml")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Config file {args.config} is empty or invalid.")
    
    # コマンドライン引数で上書き
    if args.layer:
        config["layers"] = [args.layer]
    if args.signal:
        config["signal"] = args.signal
    
    model_name = config["model_name"]
    model_type = config.get("model_type", "mamba" if "mamba" in model_name.lower() else "transformer")
    model_dir_name = model_name.split("/")[-1].replace("-", "_")
    layers = config["layers"]
    signal = config["signal"]
    devices = config.get("devices", [0])

    output_root = resolve_output_root(config)
    model_dir = output_root / model_dir_name
    copy_config(args.config, model_dir)
    
    device = torch.device(f"cuda:{devices[0]}")
    dtype = torch.float16
    
    # データセットをロード（解析用）
    dataset_config = config["dataset"]
    ds = load_dataset(dataset_config["name"], dataset_config.get("config"), split=dataset_config["split"])
    
    # モデルとトークナイザーをロード
    model, tokenizer, device_model = load_model_and_tokenizer(model_name, model_type, devices, dtype)
    
    # 各層について解析
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Analyzing SAE for layer {layer}, signal={signal}")
        print(f"{'='*60}")
        
        output_dir = model_dir / f"layer_{layer}" / signal
        checkpoint_path = output_dir / "sae_checkpoint.pt"
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # SAEをロード
        sae, checkpoint = load_sae_checkpoint(checkpoint_path, device)
        
        # データをロード
        chunk_files = sorted(output_dir.glob("chunk_*.pt"))
        if len(chunk_files) == 0:
            print(f"No chunk files found in {output_dir}")
            continue
        
        # サンプルをロード（全体をロードするとメモリ不足になる可能性があるため）
        print("Loading sample data for analysis...")
        sample_data = torch.load(chunk_files[0], map_location=device)
        if len(chunk_files) > 1:
            # 最初の2つのchunkを結合（必要に応じて調整）
            sample_data = torch.cat([
                sample_data,
                torch.load(chunk_files[1], map_location=device),
            ], dim=0)
        
        # 標準化
        mean = checkpoint["mean"].to(device)
        std = checkpoint["std"].to(device)
        sample_data_norm = (sample_data - mean) / std
        
        # 1. Sparsity分布
        print("\n1. Computing sparsity distribution...")
        global_sparsity, per_feature_sparsity, z_all = compute_sparsity_distribution(sae, sample_data_norm)
        print(f"Global sparsity: {global_sparsity:.4f}")
        
        # ヒストグラムを保存
        plt.figure(figsize=(10, 6))
        plt.hist(per_feature_sparsity.numpy(), bins=50)
        plt.xlabel("Nonzero ratio per feature")
        plt.ylabel("Count")
        plt.title(f"Sparsity distribution (global={global_sparsity:.4f})")
        plt.savefig(analysis_dir / "sparsity_distribution.png")
        plt.close()
        
        # 2. Half-life分布（state-SAEの場合のみ）
        if signal == "state" and not args.skip_half_life:
            print("\n2. Computing half-life distribution...")
            # 長いシーケンスを生成
            texts = []
            for i, ex in enumerate(ds):
                if len(texts) >= 200:
                    break
                text = ex["text"].strip()
                if text:
                    texts.append(text)
            
            long_text = " ".join(texts)
            enc_long = tokenizer(
                long_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device_model)
            
            with torch.no_grad():
                out_long = model(
                    **enc_long,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hs_long = out_long.hidden_states[layer][0].float()  # [T, d] on model device
                mean = checkpoint["mean"].to(hs_long.device)
                std = checkpoint["std"].to(hs_long.device)
                features_long_norm = (hs_long - mean) / std
                _, z_long = sae(features_long_norm)
                z_long = z_long.cpu()
            
            taus = compute_half_lives(z_long)
            print(f"Half-life stats: min={taus.min()}, median={np.median(taus):.1f}, max={taus.max()}")
            
            # ヒストグラムを保存
            plt.figure(figsize=(10, 6))
            plt.hist(taus, bins=range(0, taus.max()+2, 2))
            plt.xlabel("Approximate half-life (lags)")
            plt.ylabel("Count")
            plt.title("Feature half-life distribution")
            plt.savefig(analysis_dir / "half_life_distribution.png")
            plt.close()
        
        # 3. Top contexts
        print("\n3. Extracting top activation contexts...")
        contexts, tokens = extract_top_contexts(
            model=model,
            tokenizer=tokenizer,
            device=device_model,
            sae=sae,
            checkpoint=checkpoint,
            dataset=ds,
            target_layer=layer,
            signal=signal,
            num_features=10,
        )
        
        # 各featureのcontextsを保存
        for feature_id, feature_contexts in contexts.items():
            context_file = analysis_dir / f"contexts_feature_{feature_id}.txt"
            with open(context_file, "w", encoding="utf-8") as f:
                f.write(f"Feature {feature_id} - Top activation contexts\n")
                f.write("=" * 60 + "\n\n")
                for ctx in feature_contexts:
                    f.write(f"Rank {ctx['rank']}, pos {ctx['pos']}, activation={ctx['activation']:.4f}\n")
                    f.write(f"Context: {ctx['context']}\n\n")
        
        print(f"\nAnalysis complete! Results saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
