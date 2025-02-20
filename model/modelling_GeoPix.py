from typing import Union
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

from model.custom_llava import CustomLlavaForConditionalGeneration
from model.configuration_GeoPix import GeoPixConfig
from model.mask_predictor import MaskPredictorModel
from model.prompt_encoder import PromptEncoderModel
from model.memory import ClasswiseLearnableMemoryModel


class ImageNeck(nn.Module):
    def __init__(self, 
                 in_channels=1024, 
                 mid_channels=512, 
                 out_channels=256):
        super(ImageNeck, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv1(x)   
        x = self.upsample(x)
        x = self.conv2(x)   
        return x
    

class MultimodalMaskProjector(LlavaMultiModalProjector):
    pass


class GeoPixPretrainedModel(PreTrainedModel):
    config_class = GeoPixConfig
    base_model_prefix = "GeoPix"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MaskProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False


class GeoPixForConditinalGeneration(GeoPixPretrainedModel):
    vlm: CustomLlavaForConditionalGeneration
    mask_predictor: Union[MaskPredictorModel]
    prompt_encoder: Union[PromptEncoderModel]
    classwise_learnable_module: Union[ClasswiseLearnableMemoryModel]

    def __init__(self, config:GeoPixConfig):
        super().__init__(config)

        self.config = config

        self.vlm = CustomLlavaForConditionalGeneration

        # Initialize Segmentation Head
        self.mask_predictor = MaskPredictorModel

        # Initialize Prompt Encoder
        self.prompt_encoder = PromptEncoderModel

        # Initialize Memory Bank
        self.classwise_learnable_module = ClasswiseLearnableMemoryModel
        
        self.image_neck_rough = self._init_image_neck(config)
        self.image_neck_detail = self._init_image_neck(config)

        # segment codebook
        self.multiseg_scalar = [torch.nn.Parameter(torch.ones([]) * (1 / config.seg_token_num)) for _ in range(config.seg_token_num)]
        self.multiscale_scalar = [torch.nn.Parameter(torch.ones([]) * (1 / config.image_feature_scale_num)) for _ in range(config.image_feature_scale_num)]

        embed_dim = 4096
        temp_dim = 1024
        out_chans = 256
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_chans),
        )

        # Post-initialization
        self.post_init()

    def _init_image_neck(self, config:GeoPixConfig):
        image_neck = ImageNeck()
        return image_neck

    def inference(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        pixel_values: torch.tensor,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        outputs = self.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        hidden_states = outputs.hidden_states
        output_ids = outputs.sequences

        generated_token_num = output_ids.shape[1] - input_ids.shape[1]
        generated_ids = output_ids[:, -generated_token_num:]

        seg_token_ids = torch.tensor([32002, 32003, 32004, 32005, 32006, 32007], device="cuda")
        seg_len = len(seg_token_ids)
        seg_token_mask = torch.zeros_like(generated_ids, dtype=torch.bool)

        for i in range(len(generated_ids[0]) - seg_len + 1):
            if torch.all(generated_ids[0, i:i+seg_len] == seg_token_ids):
                seg_token_mask[0, i:i+seg_len] = True

        
        if not torch.any(seg_token_mask == True): # 没有seg token
            return generated_ids, None

        seg_token_positions = torch.where(seg_token_mask[0])[0].reshape(-1, seg_len)

        vlm_hidden_state = [hs[-1] for hs in hidden_states]
        vlm_hidden_state[0] = vlm_hidden_state[0][:, -1, :].unsqueeze(1)
        vlm_hidden_state = torch.cat(vlm_hidden_state, dim=1)
        
        vlm_hidden_state = self.text_hidden_fcs(vlm_hidden_state)

        group_hidden_states = []
        for positions in seg_token_positions:
            hidden_states = vlm_hidden_state[0, positions]
            group_hidden_states.append(hidden_states)
        group_hidden_states = torch.stack(group_hidden_states)

        group_hidden_states = group_hidden_states.view(
            len(group_hidden_states),
            self.config.image_feature_scale_num,
            self.config.seg_token_num,
            group_hidden_states.shape[-1]
        )

        fused_group_hidden_states = group_hidden_states[:, :, 0] * 0

        for i in range(self.config.seg_token_num):
            fused_group_hidden_states = fused_group_hidden_states + self.multiseg_scalar[i] * group_hidden_states[:, :, i]

        group_hidden_states = fused_group_hidden_states

        residual_image_feature = self.get_residual_image_feature(pixel_values=pixel_values)
        img_embeddings = residual_image_feature.flatten(1, 2)
        patch_count = residual_image_feature.shape[2]
        patch_size = int(patch_count ** 0.5)
        img_embeddings = img_embeddings.permute(0,2,1)
        img_embeddings = img_embeddings.view(self.config.image_feature_scale_num,-1,patch_size, patch_size)
        
        rough_input = img_embeddings[0].unsqueeze(0)
        detail_input = img_embeddings[1].unsqueeze(0)

        rough_embeddings = self.image_neck_rough(rough_input)
        detail_embeddings = self.image_neck_detail(detail_input)

        rough_embeddings = rough_embeddings.squeeze(0)
        detail_embeddings = detail_embeddings.squeeze(0)

        _img_embeddings = torch.stack([rough_embeddings, detail_embeddings], dim=0)

        class_idx = [74]
        sparse_embeddings, dense_embeddings, image_pe = self.prompt_encoder(group_hidden_states)
        sparse_embeddings = sparse_embeddings.to(group_hidden_states.dtype)

        out_size = 128
        pred_masks = []
        low_res_masks = torch.zeros([sparse_embeddings.shape[0], 1, out_size, out_size]).to(_img_embeddings)
        for l in range(self.config.image_feature_scale_num):
            img_embeds = _img_embeddings[l].unsqueeze(0)
            temp_low_res_masks, _ = self.mask_predictor(
                img_embeds=img_embeds,
                image_pe=image_pe,
                sparse_embeddings=sparse_embeddings, 
                dense_embeddings=dense_embeddings,
                previous_masks=l_low_res_masks if l > 0 else None,
                level=l,
            )

            res_masks_for_mem_enc = F.interpolate(input=temp_low_res_masks.float(), size=(out_size, out_size),mode="bilinear",align_corners=False).to(low_res_masks)

            mem_img_embeds = self.classwise_learnable_module(
                img_embeds=img_embeds,
                obj_masks=res_masks_for_mem_enc,
                class_idx=class_idx,
                img_feature_level=l
            )

            l_low_res_masks, _ = self.mask_predictor(
                img_embeds=mem_img_embeds,
                image_pe=image_pe,
                sparse_embeddings=sparse_embeddings, 
                dense_embeddings=dense_embeddings,
                previous_masks=l_low_res_masks if l >0 else None,
                level=l,
            )

            low_res_masks = low_res_masks + self.multiscale_scalar[l] * F.interpolate(l_low_res_masks.float(), (out_size, out_size),mode="bilinear",align_corners=False,).to(l_low_res_masks)

        pred_mask = low_res_masks[:, 0]
        
        return generated_ids, pred_mask
        
    def get_residual_image_feature(self, pixel_values: torch.FloatTensor):
        rough_image_feature, detail_image_feature = self.vlm.get_image_embeds(
            pixel_values=pixel_values, vision_feature_layers=[-11, -2])
        
        residual_image_feature = torch.stack([detail_image_feature, rough_image_feature], dim=0)

        return residual_image_feature        


    @classmethod
    def from_pretrained(cls, load_directory: str):
        config = cls.config_class.from_pretrained(load_directory)

        model = cls(config)
        
        model.vlm = model.vlm.from_pretrained(os.path.join(load_directory, 'vlm'), low_cpu_mem_usage=True)#, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        model.mask_predictor = model.mask_predictor.from_pretrained(os.path.join(load_directory, 'seg'), low_cpu_mem_usage=True,)
        model.prompt_encoder = model.prompt_encoder.from_pretrained(os.path.join(load_directory, 'pec'), low_cpu_mem_usage=True,)
        model.classwise_learnable_module = model.classwise_learnable_module.from_pretrained(os.path.join(load_directory, 'clm'), low_cpu_mem_usage=True,)

        rough_projector_path = os.path.join(load_directory, 'image_neck_rough.pth')
        detail_projector_path = os.path.join(load_directory, 'image_neck_detail.pth')
        model.image_neck_rough.load_state_dict(torch.load(rough_projector_path, weights_only=True))
        model.image_neck_detail.load_state_dict(torch.load(detail_projector_path, weights_only=True))

        multiseg_scalar_path = os.path.join(load_directory, 'multiseg_scalar.pth')
        multiscale_scalar_path = os.path.join(load_directory, 'multiscale_scalar.pth')
        multiseg_scalar_values = torch.load(multiseg_scalar_path, weights_only=True)
        multiscale_scalar_values = torch.load(multiscale_scalar_path, weights_only=True)

        for param, value in zip(model.multiseg_scalar, multiseg_scalar_values):
            param.data = value

        for param, value in zip(model.multiscale_scalar, multiscale_scalar_values):
            param.data = value

        text_hidden_fcs_path = os.path.join(load_directory, 'text_hidden_fcs.pth')
        model.text_hidden_fcs.load_state_dict(torch.load(text_hidden_fcs_path, weights_only=True))

        return model