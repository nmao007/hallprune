import sys
import torch
sys.path.append("./sparsegpt")
from sparsegpt import SparseGPT

class Hall_SparseGPT(SparseGPT):
    def add_iti_penalty(self, theta_direction, alpha=50.0):
        """
        Adds the rank-1 penalty (alpha * theta * theta^T) to self.H
        """
        if theta_direction is None:
            return

        theta = theta_direction.to(self.dev).float()
        
        # Ensure dimensions match Hessian shape [d_in, d_in]
        assert theta.shape[0] == self.columns, \
            f"Dimension mismatch: Probe {theta.shape[0]} vs Hessian {self.columns}"

        # Rank-1 Outer Product
        P_truth = torch.outer(theta, theta)
        
        # Add penalty directly to accumulated Hessian
        self.H += alpha * P_truth