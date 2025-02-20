import torch
from transformers import PreTrainedModel

from .block.mask_decoder_multi_scale import MaskDecoderMultiScale
from .block.transformer import TwoWayTransformer

from .configuration_mask_predictor import MaskPredictorConfig

class MaskPredictorModel(PreTrainedModel):
    config_class = MaskPredictorConfig

    def __init__(
        self, 
        config:MaskPredictorConfig
    ):
        super().__init__(config)
        
        image_feature_scale_num = config.image_feature_scale_num
        prompt_embed_dim = config.prompt_embed_dim
        mask_decoder_transformer_depth = config.mask_decoder_transformer_depth

        self.image_feature_scale_num = image_feature_scale_num

        self.mask_decoder=MaskDecoderMultiScale(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=mask_decoder_transformer_depth,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            image_feature_scale_num=image_feature_scale_num
        ) 

        self.post_init()
            
    def forward(
        self,
        img_embeds:torch.Tensor,
        image_pe:torch.Tensor,
        sparse_embeddings:torch.Tensor, 
        dense_embeddings:torch.Tensor,
        previous_masks:torch.Tensor,
        level:int,
    ):
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=img_embeds, 
            image_pe=image_pe, 
            sparse_prompt_embeddings=sparse_embeddings[:, level].unsqueeze(1), 
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=False, 
            previous_masks=previous_masks if level>0 else None, 
            level_num=level
        )

        return low_res_masks, iou_predictions

