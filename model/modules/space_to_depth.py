"""
Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15
"""
import torch
import torch.nn as nn
from base import BaseModel

class SpaceToDepth(BaseModel):
    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

    
class DepthToSpace(nn.Module):
    def __init__(self, block_size: int):
        super().__init__()
        self.bs = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x
