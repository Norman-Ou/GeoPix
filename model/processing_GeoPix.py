# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for GeoPix.
"""

from transformers import ProcessorMixin, BatchFeature
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs

IGNORE_INDEX = -100

class GeoPixProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token", "seg_token_ids"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer = None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        seg_token_ids=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token

        self.seg_token_ids = seg_token_ids
        self.seg_token_num = 3
        self.image_feature_scale_num = 2

        super().__init__(image_processor, tokenizer, chat_template=chat_template)


class GeoPixValidProcessor(GeoPixProcessor):
    def __call__(self, batch) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            padding=True, 
            return_tensors="pt", 
            padding_side="left"
        )

        cnt = 0
        offsets = [0]
        image_path_list, image_list = [], []
        conversations_list, masks_list, mask_labels_list, class_ids_list = [], [], [], []

        for item in batch:
            image_path = item['image_path']
            image = item['image']
            conversations = item['conversations']
            masks =  item["masks"]
            mask_labels = item["mask_labels"]
            class_ids = item["class_ids"]

            cnt += len(conversations)
            offsets.append(cnt)

            image_path_list.append(image_path)
            image_list.append(image)
            conversations_list.append(conversations)
            masks_list.append(masks.float())
            mask_labels_list.append(mask_labels)
            class_ids_list.append(class_ids)
        
        # Image Preprocess
        if len(image_list) > 0:
            image_inputs = self.image_processor(image_list, **output_kwargs["images_kwargs"])
        else:
            image_inputs = None

        # Text Preprocess
        prompt_strings = conversations_list
        if image_inputs is not None:
            pixel_values = image_inputs["pixel_values"]
            height, width = get_image_size(to_numpy_array(pixel_values[0]))
            num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1
            prompt_strings = []
            for conversations in conversations_list:
                for sample in conversations:
                    sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                    prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data=dict(
                **text_inputs,
                **image_inputs,
                class_ids = class_ids_list,
            )
        )








