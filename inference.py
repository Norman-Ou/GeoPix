import os
import numpy as np
import cv2
from PIL import Image
import re
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_folder = os.path.dirname(os.path.abspath(__file__))

from dataset.inference_input import InferenceInputData
from engine import GeoPixInferenceEngine

mask_colors = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 128]
], dtype=np.uint8)

examples = [
    ["Referring Segmentation", "Can you segment the vehicle on the upper right?", "./imgs/4.jpg"],
    ["Referring Segmentation", "Can you segment the storage tank in the top and the storage tank in the bottom?", "./imgs/1.jpg"],
    ["Visual Question Answering", "What object class is the vehicle on the middle of right?", "./imgs/2.jpg"],
    ["Visual Grounding", "Where is the storage tank that is a little smaller than the large storage tank?", "./imgs/2.jpg"],
    ["Caption", "Describe the image in detail.", "./imgs/3.jpg"],
]

def mask_postprocess(image_path, output_masks, mask_colors):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = image_np.astype(np.float32)
    height, width = image_np.shape[:2]
    color_mask = image_np.copy()

    if output_masks is not None:
        for i, pm in enumerate(output_masks):
            pm = pm.float().detach().cpu().numpy()
            pm = cv2.resize(pm, (width, height), interpolation=cv2.INTER_LINEAR)
            pm = pm > 0
            color_mask[pm] = color_mask[pm] * 0.5 + mask_colors[i] * 0.5

    save_img = np.clip(color_mask, 0, 255).astype(np.uint8)
    save_img_pil = Image.fromarray(save_img)
    save_path = os.path.join(current_folder, "0_referring_segmentation_output.jpg")
    save_img_pil.save(save_path)

    return save_img_pil, save_path

def bbox_postprocess(image_path, output_bboxes, bbox_colors):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = image_np.astype(np.float32)
    height, width = image_np.shape[:2]

    for i, bbox in enumerate(output_bboxes):
        x_min = int(bbox[0] / 100 * width)
        y_min = int(bbox[1] / 100 * height)
        x_max = int(bbox[2] / 100 * width)
        y_max = int(bbox[3] / 100 * height)
        color = bbox_colors[i]
        thickness = 2
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color.tolist(), thickness)

    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    save_img_pil = Image.fromarray(image_np)
    save_path = os.path.join(current_folder, "0_visual_grounding_output.jpg")
    save_img_pil.save(save_path)

    return save_img_pil, save_path


def inference(task_type, input_str, input_image):
    if task_type == "Caption":
        task_identifier = "[caption]"
        input_str = "Describe the image in detail"
    elif task_type == "Visual Grounding":
        task_identifier = "[refer]"
    elif task_type == "Visual Question Answering":
        task_identifier = "[Visual Question Answering]"
    elif task_type == "Referring Segmentation":
        task_identifier = "[segment]"
    
    question = task_identifier + " " + input_str
    input_data = InferenceInputData(question=question, image_path=input_image)
    input_batch = [input_data[0]]
    input_batch = geopix_task.valid_processor(input_batch)
    output_texts, output_masks = geopix_task.inference_step(input_batch)

    bboxes = []
    if task_type == "Visual Grounding":
        pattern = r"\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>\}"
        matches = re.findall(pattern, output_texts)
        if matches:
            bboxes = [
                (max(int(match[0]), 0), max(int(match[1]), 0), max(int(match[2]), 0), max(int(match[3]), 0))
                for match in matches
            ]

    if output_masks is not None:
        output_img, mask_path = mask_postprocess(input_image, output_masks, mask_colors)
        print(f"Reference segmentation result saved at: {mask_path}")
        output_texts = output_texts.replace("[SEG0][SEG1][SEG2][SEG3][SEG4][SEG5]", "<SEG>")
    elif len(bboxes) > 0:
        output_img, bbox_path = bbox_postprocess(input_image, bboxes, mask_colors)
        print(f"Reference segmentation result saved at: {bbox_path}")
    else:
        output_img = input_image
    
    output_texts = "ASSISTANT: " + output_texts
    return output_texts, output_img


if __name__ == "__main__":
    geopix_task = GeoPixInferenceEngine(
        pretrained_model_path="pretrained_models/GeoPix-ft-sior_rsicap",
        pretrained_processor_path="pretrained_models/GeoPix-ft-sior_rsicap",
    )

    for ex in tqdm(examples):
        output_texts, output_img = inference(ex[0],ex[1],ex[2])
        print(output_texts)