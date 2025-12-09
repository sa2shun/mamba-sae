#!/usr/bin/env python3
"""
Pairwise SAE dictionary comparison utilities.

Usage:
python analysis/compare_saes.py --ckpt-a pathA --ckpt-b pathB --top-k 50
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


def load_dictionary(path: Path) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    if "decoder.weight" in state:
        weight = state["decoder.weight"]
    elif "module.decoder.weight" in state:
        weight = state["module.decoder.weight"]
    else:
        raise KeyError("decoder.weight not found in checkpoint state_dict")
    dictionary = weight.t().float()  # [hidden_dim, input_dim]
    return {"dict": dictionary, "meta": ckpt}


def cosine_dictionary_similarity(W1: torch.Tensor, W2: torch.Tensor) -> Dict[str, float]:
    W1n = F.normalize(W1, dim=1)
    W2n = F.normalize(W2, dim=1)
    sim = torch.matmul(W1n, W2n.T)
    return {
        "mean": sim.mean().item(),
        "mean_max_rows": sim.max(dim=1).values.mean().item(),
        "mean_max_cols": sim.max(dim=0).values.mean().item(),
    }


def subspace_similarity(W1: torch.Tensor, W2: torch.Tensor, top_k: int = 50) -> Dict[str, float]:
    q = min(top_k, W1.shape[0], W2.shape[0], W1.shape[1], W2.shape[1])
    if q < 2:
        return {"mean_cos_angle": float("nan"), "min_cos_angle": float("nan")}

    _, _, V1 = torch.pca_lowrank(W1, q=q)
    _, _, V2 = torch.pca_lowrank(W2, q=q)

    M = torch.matmul(V1.T, V2)
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    return {
        "mean_cos_angle": s.mean().item(),
        "min_cos_angle": s.min().item(),
    }


def random_baseline(W: torch.Tensor, top_k: int) -> Dict[str, float]:
    rand = torch.randn_like(W)
    return {
        "cosine_mean": cosine_dictionary_similarity(W, rand)["mean"],
        "subspace_mean": subspace_similarity(W, rand, top_k=top_k)["mean_cos_angle"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two SAE dictionaries")
    parser.add_argument("--ckpt-a", type=str, required=True)
    parser.add_argument("--ckpt-b", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, help="Optional path to save JSON results")
    args = parser.parse_args()

    A = load_dictionary(Path(args.ckpt_a))
    B = load_dictionary(Path(args.ckpt_b))

    cos = cosine_dictionary_similarity(A["dict"], B["dict"])
    sub = subspace_similarity(A["dict"], B["dict"], top_k=args.top_k)
    rand_a = random_baseline(A["dict"], args.top_k)
    rand_b = random_baseline(B["dict"], args.top_k)

    result = {
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "cosine": cos,
        "subspace": sub,
        "random_baseline_a": rand_a,
        "random_baseline_b": rand_b,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
