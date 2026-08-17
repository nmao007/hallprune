import sys
import torch
from pathlib import Path
from sparsegpt import SparseGPT

class Hall_SparseGPT(SparseGPT):
    def __init__(self, layer):
            super().__init__(layer)

    def add_iti_penalty(self, theta_direction, alpha=50.0):
            if theta_direction is None:
                return

            # 1. Force everything to float32 precision
            self.H = self.H.float()
            theta = theta_direction.to(self.dev).float()
            theta = theta / torch.norm(theta, p=2)

            # 2. Scale penalty relative to Trace of Hessian
            H_trace = torch.trace(self.H)
            scale = alpha * H_trace if H_trace > 0 else alpha

            # 3. Add the penalty in float32
            self.H += scale * torch.outer(theta, theta)

            # 4. Force strict symmetry
            self.H = (self.H + self.H.T) / 2.0