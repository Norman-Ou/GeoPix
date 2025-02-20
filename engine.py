from typing import List, Union
import torch
from transformers import PreTrainedTokenizer

from model.processing_GeoPix import GeoPixValidProcessor
from model.modelling_GeoPix import GeoPixForConditinalGeneration


class GeoPixInferenceEngine():
    model: Union[GeoPixForConditinalGeneration]
    valid_processor: GeoPixValidProcessor
    valid_tokenizer: PreTrainedTokenizer
    seg_token_idx: List[int]
    
    model_max_length: int = 512
    seg_token_num: int = 3
    image_feature_scale_num: int = 2
    use_mm_start_end: bool = True

    def __init__(
            self, 
            pretrained_model_path:str,
            pretrained_processor_path:str,
    ):
        super().__init__()

        processor = GeoPixValidProcessor.from_pretrained(pretrained_processor_path)
        self.valid_processor = processor
        self.valid_tokenizer = processor.tokenizer

        model = GeoPixForConditinalGeneration.from_pretrained(
            pretrained_model_path,
        )

        vision_tower = model.vlm.vision_tower
        for p in vision_tower.parameters():
            p.requires_grad = False

        self.model = model

    def inference_step(self, batch):
        self.model.eval()
        self.model.to(device="cuda", dtype=torch.bfloat16)
        for key, value in batch.items():
            try:
                if isinstance(batch[key], torch.Tensor | torch.FloatTensor):
                    batch[key] = value.to("cuda")
                if isinstance(batch[key], list):
                    for i, item in enumerate(batch[key]):
                        batch[key][i] = item.to("cuda")
            except Exception as e:
                continue

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            generate_ids, pred_masks = self.model.inference(**batch, max_new_tokens=256)

        output_texts = self.valid_tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
        output_texts = output_texts[0].split('</s>')[0]
        
        if pred_masks is not None:
            return output_texts, pred_masks
        else:
            return output_texts, None
        