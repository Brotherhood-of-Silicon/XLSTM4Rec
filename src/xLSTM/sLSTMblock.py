import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class sLSTMblock(nn.Module):
    """
    A custom sLSTM block for sequence modeling.

    Args:
        x_example (torch.Tensor): An example input tensor used to initialize the layer dimensions.
        depth (int): The depth parameter for the BlockDiagonal gates.
        dropout (float, optional): Dropout rate for regularization. Default is 0.2.

    Attributes:
        input_size (int): The size of the input features.
        ln (nn.LayerNorm): Layer normalization applied to the input.
        conv (CausalConv1D): Causal convolution layer.
        drop (nn.Dropout): Dropout layer for regularization.
        i_gate, f_gate, o_gate, z_gate (BlockDiagonal): Input, forget, output, and cell gates.
        ri_gate, rf_gate, ro_gate, rz_gate (BlockDiagonal): Recurrent input, forget, output, and cell gates.
        ln_i, ln_f, ln_o, ln_z (nn.LayerNorm): Layer normalization for gates.
        GN (nn.LayerNorm): Layer normalization applied to the hidden state.
        ln_c, ln_n, ln_h (nn.LayerNorm): Layer normalization for cell state, input modulation, and hidden state.
        left_linear, right_linear (nn.Linear): Linear layers for the final projection.
        ln_out (nn.LayerNorm): Layer normalization before the output projection.
        proj (nn.Linear): Linear layer for projecting the output to the input size.
        nt_1, ct_1, ht_1, mt_1 (torch.Tensor): Hidden and cell states initialized as zeros.

    Methods:
        init_states(x):
            Initializes the hidden and cell states with zeros based on the input tensor dimensions.
        
        forward(x):
            Defines the forward pass of the sLSTM block. Applies layer normalization, causal convolution,
            gate operations, and combines the outputs through linear transformations and layer normalization.
    
    Example:
        model = sLSTMblock(x_example=torch.randn(1, 8, 16), depth=4)
        model.init_states(torch.randn(1, 8, 16))
        output = model(torch.randn(1, 8, 16))
    """
    def __init__(self, x_example, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        conv_channels = x_example.shape[1]
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.conv = CausalConv1D(self.input_size, self.input_size, int(self.input_size/8))
        self.drop = nn.Dropout(dropout)
        
        self.i_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.f_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.o_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.z_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        
        self.ri_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rf_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.ro_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rz_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)

        self.ln_i = nn.LayerNorm(self.input_size)
        self.ln_f = nn.LayerNorm(self.input_size)
        self.ln_o = nn.LayerNorm(self.input_size)
        self.ln_z = nn.LayerNorm(self.input_size)
        
        self.GN = nn.LayerNorm(self.input_size)
        self.ln_c = nn.LayerNorm(self.input_size)
        self.ln_n = nn.LayerNorm(self.input_size)
        self.ln_h = nn.LayerNorm(self.input_size)
        
        self.left_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))
        self.right_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))

        self.ln_out = nn.LayerNorm(int(self.input_size*(4/3)))
        
        self.proj = nn.Linear(int(self.input_size*(4/3)), self.input_size)
        
        self.init_states(x_example)
        
    def init_states(self, x):
        self.nt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ct_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ht_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.mt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        
    def forward(self, x):
        x = self.ln(x)
        
        x_conv = F.silu( self.drop(self.conv( x.transpose(1, 2) ).transpose(1, 2) ) )
        
        # start sLSTM
        ht_1 = self.ht_1
        
        i = torch.exp(self.ln_i( self.i_gate(x_conv) + self.ri_gate(ht_1) ) )
        f = torch.exp( self.ln_f(self.f_gate(x_conv) + self.rf_gate(ht_1) ) )

        m = torch.max(torch.log(f)+self.mt_1[:, 0, :].unsqueeze(1), torch.log(i))
        i = torch.exp(torch.log(i) - m)
        f = torch.exp(torch.log(f) + self.mt_1[:, 0, :].unsqueeze(1)-m)
        self.mt_1 = m.detach()
        
        o = torch.sigmoid( self.ln_o(self.o_gate(x) + self.ro_gate(ht_1) ) )
        z = torch.tanh( self.ln_z(self.z_gate(x) + self.rz_gate(ht_1) ) )
        
        ct_1 = self.ct_1
        ct = f*ct_1 + i*z
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        nt_1 = self.nt_1
        nt = f*nt_1 + i
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        
        ht = o*(ct/nt) # torch.Size([4, 8, 16])
        ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
        self.ht_1 = ht.detach()
        # end sLSTM
        
        slstm_out = self.GN(ht)
        
        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))
        
        out = self.ln_out(left*right)
        out = self.proj(out)
        return out
  
