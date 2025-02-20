import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class Conv3dFuser(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            out_channels: int = 1,
    ):  
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1),
            nn.BatchNorm3d(1),
        )
    def forward(self, memory:torch.Tensor):
        memory = memory.permute(2, 1, 0, 3, 4)
        fused_memory = self.model(memory)
        fused_memory = fused_memory.permute(2, 1, 0, 3, 4).squeeze(1)
        return fused_memory