from transformers import PretrainedConfig

class MaskPredictorConfig(PretrainedConfig):
    model_type = "mask_predictor"

    def __init__(
        self,
        image_feature_scale_num: int = 2,
        prompt_embed_dim: int = 256,
        mask_decoder_transformer_depth: int = 2,
        **kwargs
    ):
        self.image_feature_scale_num = image_feature_scale_num
        self.prompt_embed_dim = prompt_embed_dim
        self.mask_decoder_transformer_depth = mask_decoder_transformer_depth
        
        super().__init__(**kwargs)