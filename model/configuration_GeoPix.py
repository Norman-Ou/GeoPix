import os
import json
from typing import Optional, Union, Dict

from transformers import PretrainedConfig
from transformers.models.sam import SamConfig
from transformers.models.llava import LlavaConfig

from model.memory import ClasswiseLearnableMemoryConfig
from model.prompt_encoder import PromptEncoderConfig
from model.mask_predictor import MaskPredictorConfig

class GeoPixConfig(PretrainedConfig):
    model_type = "GeoPix"

    def __init__(
        self,
        vlm_config: Optional[Union[LlavaConfig, Dict]] = None,
        seg_config: Optional[Union[MaskPredictorConfig, Dict]] = None,
        clm_config: Optional[Union[ClasswiseLearnableMemoryConfig, Dict]] = None,
        pec_config: Optional[Union[PromptEncoderConfig, Dict]] = None,
        seg_token_num: int = 3,
        image_feature_scale_num: int = 2,
        ce_loss_weight:float = 1.0,
        bce_loss_weight:float = 2.0,
        dice_loss_weight:float = 0.5,
        **kwargs,
    ):
        # Pretrain model configs
        self.vlm_config = vlm_config
        self.seg_config = seg_config
        self.clm_config = clm_config
        self.pec_config = pec_config

        self.seg_token_num = seg_token_num
        self.image_feature_scale_num = image_feature_scale_num

        self.ce_loss_weight = ce_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight


        super().__init__(**kwargs)


    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(os.path.join(load_directory, 'config.json'), 'r') as file:
            geopix_args = json.load(file) 
            
        config = cls(
            vlm_config=LlavaConfig.from_json_file(os.path.join(load_directory, 'vlm', 'config.json')),
            seg_config=MaskPredictorConfig.from_json_file(os.path.join(load_directory, 'seg', 'config.json')),
            clm_config=ClasswiseLearnableMemoryConfig.from_json_file(os.path.join(load_directory, 'clm', 'config.json')),
            pec_config=PromptEncoderConfig.from_json_file(os.path.join(load_directory, 'pec', 'config.json')),
            **geopix_args
        )
        
        return config

