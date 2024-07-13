import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class mLSTMblock(nn.Module):
    """
    A custom mLSTM block for sequence modeling with multi-head attention and gated operations.

    Args:
        x_example (torch.Tensor): An example input tensor used to initialize the layer dimensions.
        factor (float): The factor to determine the hidden size relative to the input size.
        depth (int): The depth parameter for the BlockDiagonal gates.
        dropout (float, optional): Dropout rate for regularization. Default is 0.2.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layer, determined by the input size and factor.
        ln (nn.LayerNorm): Layer normalization applied to the input.
        left, right (nn.Linear): Linear layers for splitting the input.
        conv (CausalConv1D): Causal convolution layer.
        drop (nn.Dropout): Dropout layer for regularization.
        lskip (nn.Linear): Linear layer for skip connection.
        wq, wk, wv (BlockDiagonal): Linear layers for query, key, and value in multi-head attention.
        dropq, dropk, dropv (nn.Dropout): Dropout layers for query, key, and value.
        i_gate, f_gate, o_gate (nn.Linear): Input, forget, and output gates.
        ln_c, ln_n (nn.LayerNorm): Layer normalization for cell state and input modulation.
        lnf, lno, lni (nn.LayerNorm): Layer normalization for forget, output, and input gates.
        GN (nn.LayerNorm): Layer normalization applied to the hidden state.
        ln_out (nn.LayerNorm): Layer normalization before the output projection.
        drop2 (nn.Dropout): Dropout layer for regularization.
        proj (nn.Linear): Linear layer for projecting the output to the input size.
        ln_proj (nn.LayerNorm): Layer normalization after the output projection.
        ct_1, nt_1 (torch.Tensor): Hidden and cell states initialized as zeros.

    Methods:
        init_states(x_example):
            Initializes the hidden and cell states with zeros based on the input tensor dimensions.
        
        forward(x):
            Defines the forward pass of the mLSTM block. Applies layer normalization, 
            linear transformations, causal convolution, multi-head attention, gated operations,
            and combines the outputs through linear transformations and layer normalization.
    
    Example:
        model = mLSTMblock(x_example=torch.randn(1, 8, 16), factor=2, depth=4)
        model.init_states(torch.randn(1, 8, 16))
        output = model(torch.randn(1, 8, 16))
    """
    def __init__(self, x_example, factor, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        self.hidden_size = int(self.input_size*factor)
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
        self.drop = nn.Dropout(dropout+0.1)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.dropq = nn.Dropout(dropout/2)
        self.dropk = nn.Dropout(dropout/2)
        self.dropv = nn.Dropout(dropout/2)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)
        
        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)
        
        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(dropout)
        
        self.proj = nn.Linear(self.hidden_size, self.input_size)
        self.ln_proj = nn.LayerNorm(self.input_size)
        
        self.init_states(x_example)
    
    def init_states(self, x_example):
        self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
        self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
    
    def forward(self, x):
        assert x.ndim == 3
        
        x = self.ln(x) # layer norm on x
        
        left = self.left(x) # part left 
        right = F.silu(self.right(x)) # part right with just swish (silu) function

        left_left = left.transpose(1, 2)
        left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
        l_skip = self.lskip(left_left)

        # start mLSTM
        q = self.dropq(self.wq(left_left))
        k = self.dropk(self.wk(left_left))
        v = self.dropv(self.wv(left))
        
        i = torch.exp(self.lni(self.i_gate(left_left)))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        ct_1 = self.ct_1
        ct = f*ct_1 + i*v*k
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i*k
        nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        
        ht = o * ((ct*q) / torch.max(nt*q)) # [batchs_size, ?, hiddden_size]
        # end mLSTM
        ht = ht
        
        left = self.drop2(self.GN(ht + l_skip))
        
        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))
        
        return out
