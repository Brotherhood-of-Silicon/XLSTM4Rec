import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.mLSTMblock import mLSTMblock
from xLSTM.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    """
    A custom LSTM-based neural network module that combines sLSTM and mLSTM layers.

    Args:
        layers (list of str): A list specifying the types of layers to include in the model. 
                              Use 's' for sLSTMblock and 'm' for mLSTMblock.
        x_example (torch.Tensor): An example input tensor used to initialize the layers.
        depth (int, optional): The depth parameter for the sLSTM and mLSTM blocks. Default is 4.
        factor (int, optional): The factor parameter for the mLSTM block. Default is 2.
    
    Raises:
        ValueError: If an invalid layer type is specified in the `layers` list.
    
    Attributes:
        layers (nn.ModuleList): A list of initialized LSTM blocks (sLSTMblock or mLSTMblock).

    Methods:
        init_states(x):
            Initializes the states for all LSTM blocks in the model.
        
        forward(x):
            Defines the forward pass of the model. Passes the input tensor through each LSTM block
            and combines the outputs with the original input tensor.

    Example:
        model = xLSTM(layers=['s', 'm'], x_example=torch.randn(1, 10))
        model.init_states(torch.randn(1, 10))
        output = model(torch.randn(1, 10))
    """
    def __init__(self, layers, x_example, depth=4, factor=2):
        super(xLSTM, self).__init__()

        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMblock(x_example, depth)
            elif layer_type == 'm':
                layer = mLSTMblock(x_example, factor, depth)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        x_original = x.clone()
        for l in self.layers:
             x = l(x) + x_original

        return x
