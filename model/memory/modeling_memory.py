from typing import Union, List
import torch
from transformers import PreTrainedModel

from .block.learnable_memory import LearnableMemory
from .block.memory_attention import MemoryAttention, MemoryAttentionLayer
from .block.memory_encoder import MemoryEncoder, VisionSampler, CXBlock
from .block.memory_fuser import Conv3dFuser

from .configuration_memory import ClasswiseLearnableMemoryConfig

class ClasswiseLearnableMemoryModel(PreTrainedModel):
    config_class = ClasswiseLearnableMemoryConfig

    def __init__(self, config:ClasswiseLearnableMemoryConfig):
        super().__init__(config)

        memory_dim = config.memory_dim
        memory_len = config.memory_len
        num_classes = config.num_class
        num_level = config.num_level
        feature_h = config.feature_h
        feature_w = config.feature_w

        memory_fuser_type = config.memory_fuser_type
        output_dim = config.output_dim

        self.memory_encoder = MemoryEncoder(
            out_dim=memory_dim,
            mask_downsampler=VisionSampler(),
            fuser=CXBlock(
                dim=memory_dim,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True,
            ),
        )

        self.learnable_memory = LearnableMemory(
            num_classes=num_classes,
            num_level=num_level,
            memory_dim=memory_dim,
            memory_len=memory_len,
            feature_h=feature_h,
            feature_w=feature_w,
        )

        memory_fuser_cls = Conv3dFuser

        self.memory_fuser_type = memory_fuser_type
        self.memory_fuser = memory_fuser_cls()

        self.memory_attention=MemoryAttention(
            d_model=output_dim,
            layer=MemoryAttentionLayer(
                activation="relu",
                dim_feedforward=512,
                d_model=output_dim,
                kv_dim=memory_dim,
            ),
            num_layers=1,
        )

        self.post_init()

    def forward(self, img_embeds:torch.Tensor, obj_masks:torch.Tensor, class_idx:Union[int, List[int]], img_feature_level:int):
        maskmem_features = self.memory_encoder(img_embeds, obj_masks)

        memory = self.learnable_memory(class_idx=class_idx,level=img_feature_level)
        
        fused_memory = self.memory_fuser(memory)

        B,C,H,W = img_embeds.shape
        fused_memory = fused_memory.flatten(2).permute(2, 0, 1)
        img_embeds = img_embeds.flatten(2).permute(2, 0, 1)
        mem_img_embeds = self.memory_attention(curr=img_embeds, memory=fused_memory)
        mem_img_embeds = mem_img_embeds.permute(1, 2, 0).view(B, C, H, W)

        return mem_img_embeds
    

    