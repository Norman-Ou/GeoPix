from typing import List
import os
import torch

from transformers import AutoConfig
from transformers.models.llava import LlavaForConditionalGeneration

class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_embeds(self, pixel_values: torch.FloatTensor, vision_feature_layers: List[int] = [-11, -2]):
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        rough_image_feature = image_outputs.hidden_states[vision_feature_layers[0]]
        detail_image_feature = image_outputs.hidden_states[vision_feature_layers[1]]

        return rough_image_feature[:, 1:], detail_image_feature[:, 1:]
    

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str, **kwargs):
        vision_config = AutoConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, 'vision'))
        text_config = AutoConfig.from_pretrained(os.path.join(pretrained_model_name_or_path, 'text'))

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        config.vision_config = vision_config
        config.text_config = text_config

        return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, config=config)
