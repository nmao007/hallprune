import sys
import torch
from pathlib import Path

# Add the 'sparsegpt' subfolder directly to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sparsegpt_path = parent_dir / "wanda" / "lib"
sys.path.append(str(sparsegpt_path))

try:
    from sparsegpt import SparseGPT
except ImportError:
    from sparsegpt.sparsegpt import SparseGPT

class Hall_SparseGPT(SparseGPT):
    def add_iti_penalty(self, theta_direction, alpha=50.0):
            if theta_direction is None:
                return

            theta = theta_direction.to(self.dev).float()
            theta = theta / torch.norm(theta, p=2)

            # Scale penalty relative to Trace of Hessian
            H_trace = torch.trace(self.H)
            scale = alpha * H_trace if H_trace > 0 else alpha

            # Add the penalty, matching the Hessian's data type
            self.H += (scale * torch.outer(theta, theta)).to(self.H.dtype)
            
            # 🚨 THE FIX: Force strict symmetry to survive FP16 rounding errors
            self.H = (self.H + self.H.T) / 2.0