import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import repeat_modules
# from base import BaseModel

def repeat_modules(*modules: list[tuple[any, list[any]]], repeat: int = 1) -> nn.Sequential:
    """
    Given a list of modules and their parameters 
    """
    layers = []
    for _ in range(repeat):
        for module, params in modules:
            layers.append(module(*params))
    return nn.Sequential(*layers)

class KernelPrediction(nn.Module):
    """
    It is stated that the kernels are predicted in order to re-align the inputs dependent on jittering
    as such, rather than predicting a 
        out x in x kernel_size x kernel_size
    kernel, we predict a
        kernel_size x kernel_size
    """
    def __init__(self, layers: int, hidden_features: int, kernel_size: int):
        assert layers > 2
        super().__init__()

        self.kernel_size = kernel_size

        # n-layer mlp with k-hidden_features?
        net_layers = repeat_modules(
                (nn.Linear, [hidden_features, hidden_features]),
                (nn.ReLU, []),
                repeat=layers - 2
            )
        
        # assert not the same instances
        assert net_layers[0] is not net_layers[2]

        self.net = nn.Sequential(
            nn.Linear(2, hidden_features),
            nn.ReLU(),
            *net_layers,
            nn.Linear(hidden_features, kernel_size * kernel_size), # predict a 1 x 1 x 3 x 3 kernel
            nn.ReLU(),
        )   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        
        kernel = x.view(1, 1, self.kernel_size, self.kernel_size)
        return kernel
    
if __name__ == "__main__":
    net = KernelPrediction(7, 1024, 3)
    x = torch.randn(1, 2)
    kernel = net(x)
    print(kernel)
    kernel = kernel.repeat(3, 3, 1, 1)
    
    fake_img = torch.randn(1, 3, 256, 256)
    out = F.conv2d(fake_img, kernel, padding=1)
    loss = (out - fake_img).pow(2).mean()
    loss.backward()
    print(loss)
    print(out.grad)
    # print(kernel.grad)


    