from transformers import PreTrainedModel

from .block.prompt_encoder import PromptEncoder
from .configuration_prompt_encoder import PromptEncoderConfig

class PromptEncoderModel(PreTrainedModel):
    config_class = PromptEncoderConfig

    def __init__(
        self,
        config:PromptEncoderConfig
    ):
        super().__init__(config)
        prompt_embed_dim = config.prompt_embed_dim
        vit_patch_size = 14
        image_size = config.resize_vision_tower_size
        image_embedding_size = image_size // vit_patch_size

        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
    
    def forward(self, pred_embedding):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=None,masks=None,text_embeds=pred_embedding) #sparse_embeddings:N, Lev, 256
        image_pe = self.prompt_encoder.get_dense_pe()
        return sparse_embeddings, dense_embeddings, image_pe


