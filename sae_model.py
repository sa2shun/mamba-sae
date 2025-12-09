"""
SAE (Sparse Autoencoder) モデルの実装
L1正則化とk-sparseの両方に対応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """Sparse Autoencoder
    
    Args:
        input_dim: 入力次元
        hidden_dim: 隠れ層の次元（オーバーコンプリート）
        mode: "l1" or "k_sparse"
        l1_lambda: L1正則化の係数（mode="l1"の場合）
        k_frac: 活性化する特徴の割合（mode="k_sparse"の場合）
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, mode: str = "l1", l1_lambda: float = 1e-3, k_frac: float = 0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.mode = mode
        self.l1_lambda = l1_lambda
        self.k_frac = k_frac
        
    def forward(self, x):
        """Forward pass
        
        Returns:
            x_hat: 再構成された入力
            z: 活性化ベクトル
        """
        z = F.relu(self.encoder(x))
        
        if self.mode == "k_sparse":
            # k-sparse: top-kのみを残す
            k = max(1, int(self.k_frac * z.shape[-1]))
            topk_values, topk_indices = torch.topk(z, k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_indices, topk_values)
            z = z_sparse
        
        x_hat = self.decoder(z)
        return x_hat, z
    
    def loss(self, x):
        """損失関数を計算
        
        Returns:
            total_loss: 合計損失
            recon_loss: 再構成損失
            reg_loss: 正則化損失
        """
        x_hat, z = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        
        if self.mode == "l1":
            reg_loss = z.abs().mean()
            total_loss = recon_loss + self.l1_lambda * reg_loss
        elif self.mode == "k_sparse":
            # k-sparseの場合は正則化項なし（スパース性はtop-kで強制）
            reg_loss = torch.tensor(0.0, device=x.device)
            total_loss = recon_loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return total_loss, recon_loss.detach(), reg_loss.detach()
    
    def compute_sparsity(self, z, threshold: float = 1e-3):
        """スパース性を計算
        
        Args:
            z: 活性化ベクトル [N, hidden_dim]
            threshold: 閾値
            
        Returns:
            global_sparsity: 全体のスパース性（非ゼロ率）
            per_feature_sparsity: 特徴ごとのスパース性 [hidden_dim]
        """
        with torch.no_grad():
            active = (z > threshold).float()
            global_sparsity = active.mean().item()
            per_feature_sparsity = active.mean(dim=0)  # [hidden_dim]
        return global_sparsity, per_feature_sparsity

