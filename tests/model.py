import torch
import unittest
from model.model import Warping
from model.modules import DepthToSpace, SpaceToDepth
from utils import retrieve_elements_from_indices

class TestWarping(unittest.TestCase):
    def test_dilation(self):
        kernel_size = 7
        max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, return_indices=True)
        
        depth = torch.arange(1, 17).view(1, 1, 4, 4).float()
        depth = torch.cat((depth, -depth), dim=0)

        motion_x = torch.arange(1, 17).view(1, 1, 4, 4).float()
        motion_y = torch.arange(-16,0).view(1, 1, 4, 4).float()

        motion = torch.cat([motion_x, motion_y], dim=1)
        motion = motion.repeat(2, 1, 1, 1)
        
        _, indices = max_pool(depth)
        dilated_motion = retrieve_elements_from_indices(motion, indices)

        B, _, H, W = dilated_motion.shape
        for batch in range(B):
            for h in range(H):
                for w in range(W):
                    index = indices[batch, 0, h, w]
                    assert dilated_motion[batch, :, h, w].equal( motion[batch, :, index // W, index % W])

    # # not really useful without data
    # def test_warping(self):
    #     warp = Warping(scale_factor=2, depth_block_size=3)
    #     d2s =  DepthToSpace(block_size=2)

    #     current_jitter = torch.tensor([[[[0]], [[0]]]]).float()
    #     previous_jitter = torch.tensor([[[[0]], [[0]]]]).float()
    #     depth = torch.arange(1, 17).view(1, 1, 4, 4).float()
    #     motion = torch.linspace(-1, 1, 32).view(1, 2, 4, 4)
    #     prev_features = torch.arange(1, 17).view(1, 1, 4, 4).float()
    #     prev_color = torch.arange(1, 49).view(1, 3, 4, 4).float()
        
    #     features, color = warp(depth, current_jitter, previous_jitter, motion, prev_features, prev_color)
    #     print(d2s(features))
    #     print(d2s(color))

    def test_space_to_depth_and_depth_to_space(self):
        d2s = DepthToSpace(block_size=2)
        s2d = SpaceToDepth(block_size=2)

        x = torch.arange(1, 17).view(1, 1, 4, 4).float()

        y = s2d(x)
        z = d2s(y)

        assert x.equal(z)


