import torch
from torch.utils.data import Dataset
import cv2

import dataset.conversation as conversation_lib
DEFAULT_IMAGE_TOKEN = "<image>"

class VisionLanguageDataset(Dataset):
    ignore_label = 255
    img_size = 1024

    data = []

    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        ori_size = (image.shape[0], image.shape[1])
        return image, ori_size
    
    def preprocess_multimodal(self, source):
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
        return source
        

class InferenceInputData(VisionLanguageDataset):
    def __init__(
            self,
            question,
            image_path,
        ) -> None:
        super().__init__()

        self.question = question
        self.image_path = image_path
                
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        qs = self.question
        image_path = self.image_path
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

        image, ori_size = self.load_image(image_path)
        
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        conversation = [prompt]

        class_idx = [74]

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        return dict(
            image_path=image_path,
            image=image,
            conversations=conversation,
            masks=masks,
            mask_labels=label,
            class_ids=class_idx,
        )
