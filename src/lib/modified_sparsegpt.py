import torch
from .sparsegpt import SparseGPT

class Hall_SparseGPT(SparseGPT):
    def __init__(self, layer):
            super().__init__(layer)
            # mean(diag(H)) before the penalty; used so damping matches plain SparseGPT
            self.damp_base = None

    def add_truthfulness_penalty(self, theta_direction, alpha=1.0, scale_mode="mean_diag"):
            """
            Adds alpha * s * theta theta^T to the Hessian, penalizing ||dW theta||^2.

            scale_mode="mean_diag": s = mean(diag(H)), so alpha is relative to a typical
                diagonal entry and means the same thing for every layer width.
            scale_mode="trace": s = trace(H) = d * mean(diag(H)) (the original behaviour;
                alpha=50 here is ~2e5x the mean diagonal for d=4096, ~7e5x for d=14336).
            """
            if theta_direction is None:
                return

            # 1. Force everything to float32 precision
            self.H = self.H.float()
            theta = theta_direction.to(self.dev).float()
            theta_norm = torch.norm(theta, p=2)
            if theta_norm < 1e-6:
                return
            theta = theta / theta_norm

            # 2. Scale penalty relative to the (unpenalized) Hessian
            mean_diag = torch.mean(torch.diag(self.H))
            self.damp_base = mean_diag
            if scale_mode == "mean_diag":
                ref = mean_diag
            elif scale_mode == "trace":
                ref = torch.trace(self.H)
            else:
                raise ValueError(f"Unknown scale_mode {scale_mode}")
            scale = alpha * ref if ref > 0 else alpha

            # 3. Add the penalty in float32
            self.H += scale * torch.outer(theta, theta)

            # 4. Force strict symmetry
            self.H = (self.H + self.H.T) / 2.0

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01):
            # Damp relative to the pre-penalty Hessian, otherwise the penalty itself
            # inflates mean(diag(H)) and the damping by a factor of (1 + alpha * ref / tr(H)).
            return super().fasterprune(sparsity, prune_n=prune_n, prune_m=prune_m, blocksize=blocksize,
                                       percdamp=percdamp, damp_base=self.damp_base)
