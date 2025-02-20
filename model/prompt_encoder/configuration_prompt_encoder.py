from transformers import PretrainedConfig

class PromptEncoderConfig(PretrainedConfig):
    model_type = "prompt_encoder"

    def __init__(
        self,
        prompt_embed_dim:int = 256,
        vit_patch_size:int = 14,
        resize_vision_tower_size:int = 448,
        **kwargs
    ):
        self.prompt_embed_dim = prompt_embed_dim
        self.vit_patch_size = vit_patch_size
        self.resize_vision_tower_size = resize_vision_tower_size

        super().__init__(**kwargs)