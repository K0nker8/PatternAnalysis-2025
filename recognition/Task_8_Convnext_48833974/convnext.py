import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)




class Block(nn.Module):
    
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x + residual
    
N, C, H, W = 2, 8, 32, 32
x = torch.randn(N, C, H, W)
block = Block(dim=C)
out = block(x)

assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
print("Block forward pass works and output shape is correct!")

class ConvNeXt(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans = 1, num_classes = 2,
                  depths = [3, 3, 9, 3], dims = [96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        stem = (nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), nn.LayerNorm(dims[0], eps=1e-6)))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6), 
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i]) for j in range(depths[i])])
            self.stages.append(stage)
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    
    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x
    

