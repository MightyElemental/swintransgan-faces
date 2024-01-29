import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff = self.feed_forward(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x

class TransformLayer(nn.Module):
    # d_model is the number of channels and controls for the input vector size
    def __init__(self, d_model:int=512, nhead:int=4, layers:int=2):
        super(TransformLayer, self).__init__()
        mlp_ratio = 4
        #print("MLP: ",d_model*mlp_ratio)

        if nhead <= 0:
            nhead = max(d_model//4, 4) # TODO: test this
            print(f"Layer of d_model {d_model} has {nhead} heads")

        tel = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, activation="gelu", norm_first=True, dim_feedforward=d_model*mlp_ratio)
        self.trans1 = nn.TransformerEncoder(tel, layers)

    def forward(self, x):
        x = self.trans1(x)
        return x

def upscale(x:torch.Tensor, H:int, W:int):
    B, HW, C = x.shape
    assert HW == H*W
    x = x.permute(0, 2, 1) # BxCx(HxW)
    x = x.view(B, C, H, W) # BxCxHxW
    x = F.pixel_shuffle(x, 2)
    B, C, H, W = x.shape
    x = x.view(B, C, H*W) # BxCx(HxW)
    x = x.permute(0,2,1) # Bx(HxW)xC
    return x, H, W


class Mlp(nn.Module):
    """Consists of two Linear layers and ReLU activation with dropouts

    Args:
        in_features (int): The number of features as input
        hidden_features (int): The number of intermediate features (default=in_features)
        out_features (int): The number of features as output (default=in_features)
        dropout (int): The dropout rate (default=0.1)
    """
    def __init__(self, in_features:int, hidden_features:int=None, out_features:int=None, dropout:float=0.1):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.main = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.main(x)

class PosEncoder(nn.Module):
    def __init__(self, seq_len:int, d_model:int):
        super(PosEncoder, self).__init__()

        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        # Same as (10000/d_model)^-(2i) where i is the vector
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model)))
        pos_enc = torch.zeros((seq_len, d_model), dtype=torch.float32)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pos_enc.unsqueeze(0)
        
    def forward(self, x):
        B = x.shape[0]
        return x + self.pos_enc.expand(B, -1, -1).to(x.device)
