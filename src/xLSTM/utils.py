"""
Original code from :
https://github.com/akaashdash
(just for this two class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockDiagonal(nn.Module):
    """
    A linear layer with block diagonal structure.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        num_blocks (int): The number of blocks in the block diagonal structure.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        num_blocks (int): The number of blocks in the block diagonal structure.
        blocks (nn.ModuleList): A list of linear layers representing the blocks in the block diagonal structure.

    Methods:
        forward(x):
            Defines the forward pass of the block diagonal linear layer. Applies each block to the input
            and concatenates the outputs along the last dimension.
    
    Example:
        block_diag = BlockDiagonal(in_features=64, out_features=128, num_blocks=4)
        output = block_diag(torch.randn(32, 64))
    """
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks

        assert out_features % num_blocks == 0
        
        block_out_features = out_features // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, block_out_features, bias=bias)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x
    
class CausalConv1D(nn.Module):
    """
    A 1D convolutional layer with causal padding.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        dilation (int, optional): The spacing between kernel elements. Default is 1.
        kwargs (dict, optional): Additional keyword arguments for the nn.Conv1d layer.

    Attributes:
        padding (int): The amount of padding added to the input tensor to ensure causality.
        conv (nn.Conv1d): The 1D convolutional layer.

    Methods:
        forward(x):
            Defines the forward pass of the causal convolutional layer. Applies the convolution
            and removes the extra padding to maintain the causal structure.
    
    Example:
        causal_conv = CausalConv1D(in_channels=16, out_channels=32, kernel_size=3)
        output = causal_conv(torch.randn(32, 16, 100))
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]
