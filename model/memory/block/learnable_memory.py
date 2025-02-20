import torch 
import torch.nn as nn

class LearnableMemory(nn.Module):
    def __init__(
            self,
            num_classes,
            num_level,
            memory_dim, 
            memory_len,
            feature_h, 
            feature_w,
    ):
        super(LearnableMemory, self).__init__()

        self.memory = nn.Parameter(torch.randn(num_classes, num_level, memory_len, memory_dim, feature_h, feature_w), requires_grad=True)

    def forward(self, class_idx, level: int):
        if isinstance(class_idx, int):
            class_idx = [class_idx]
        memory = self.memory[class_idx, level, :, :, :]
        return memory 