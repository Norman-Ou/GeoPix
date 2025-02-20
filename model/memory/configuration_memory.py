from transformers import PretrainedConfig
from typing import Literal


class ClasswiseLearnableMemoryConfig(PretrainedConfig):
    model_type = "clm"

    def __init__(
        self,
        num_class:int = 75,
        num_level:int = 2,
        memory_len: int = 64,
        memory_dim: int = 64,
        feature_h: int = 32,
        feature_w: int = 32,

        output_dim: int = 256,

        memory_fuser_type = 'conv3d',

        **kwargs
    ):
        self.num_class = num_class
        self.num_level = num_level
        self.memory_len = memory_len
        self.memory_dim = memory_dim
        self.feature_h = feature_h
        self.feature_w = feature_w

        self.output_dim = output_dim

        self.memory_fuser_type = memory_fuser_type

        super().__init__(**kwargs)