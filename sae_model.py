"""
Sparse Autoencoder with flexible sparsity regularization.

Supports:
- k-sparse (top-k) gating
- optional L1 penalty (can be combined with k-sparse)
- firing-rate equalization penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """Sparse Autoencoder.

    Args:
        input_dim: 入力次元
        hidden_dim: 隠れ層の次元（オーバーコンプリート）
        k_frac: top-kに用いる割合（Noneまたは0で無効）
        l1_lambda: L1正則化の係数
        eq_alpha: firing-rate equalizationの係数
        eq_target: target firing rate (デフォルト: k_frac)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k_frac: float | None = 0.0,
        l1_lambda: float = 0.0,
        eq_alpha: float = 0.0,
        eq_target: float | None = None,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k_frac = k_frac if k_frac is not None else 0.0
        self.l1_lambda = l1_lambda
        self.eq_alpha = eq_alpha
        self.eq_target = eq_target if eq_target is not None else self.k_frac

    def encode(self, x):
        """Return sparsified hidden codes."""
        z = F.relu(self.encoder(x))
        if self.k_frac and self.k_frac > 0:
            k = max(1, int(self.k_frac * z.shape[-1]))
            topk_values, topk_indices = torch.topk(z, k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_indices, topk_values)
            z = z_sparse
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x):
        """Compute reconstruction + regularization losses.

        Returns:
            total_loss, recon_loss, l1_loss, eq_loss (all detached except total)
        """
        x_hat, z = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)

        l1_loss = z.abs().mean() if self.l1_lambda and self.l1_lambda > 0 else torch.tensor(
            0.0, device=x.device
        )

        eq_loss = torch.tensor(0.0, device=x.device)
        if self.eq_alpha and self.eq_alpha > 0 and self.eq_target is not None:
            firing = z.mean(dim=0)
            eq_loss = self.eq_alpha * F.mse_loss(firing, torch.full_like(firing, self.eq_target))

        total_loss = recon_loss + self.l1_lambda * l1_loss + eq_loss
        return total_loss, recon_loss.detach(), l1_loss.detach(), eq_loss.detach()

    def compute_sparsity(self, z, threshold: float = 1e-4):
        """スパース性を計算."""
        with torch.no_grad():
            active = (z > threshold).float()
            global_sparsity = active.mean().item()
            per_feature_sparsity = active.mean(dim=0)
        return global_sparsity, per_feature_sparsity

    def num_active_per_sample(self, z, threshold: float = 1e-4):
        """Return active feature counts per sample."""
        with torch.no_grad():
            active = (z > threshold).sum(dim=-1)
        return active
