import torch
import torch.nn as nn
from base import BaseModel

class Blending(BaseModel):
    def forward(self, blending_mask: torch.Tensor, current_frame: torch.Tensor, previous_frame: torch.Tensor) -> torch.Tensor:
        # I think this has to be elementwise
        return blending_mask * current_frame + (1 - blending_mask) * previous_frame