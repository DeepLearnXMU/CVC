from transformers import SamModel, SamProcessor
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import random
import cv2
import os

import torch
import json
import glob
import math
import os
from transformers import set_seed


seed = 0

output_dir = f"./experiments/masked-images"
# output_dir = "./examples"

# os.makedirs(output_dir)

set_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge", device_map=device).eval()


def get_samples():
    file_dir = "./experiments/entity_high-causality"
    file_name = glob.glob(os.path.join(file_dir, "*"))

    image_path = []
    entity = []
    for _file_name in tqdm(file_name):
        with open(_file_name) as f:
            file_str = f.read()
            json_objects = file_str.split('}\n{')
            for obj in json_objects:
                obj = obj.strip()
                if not obj.startswith('{'):
                    obj = '{' + obj
                if not obj.endswith('}'):
                    obj = obj + '}'
                _entity = json.loads(obj)
            
                image_path.append(_entity["image_path"])
                entity.append(_entity)
    
    return image_path, entity


def mask_entity(image_path, entity_item, output_path):
    bbox = entity_item["bbox"]
    entity = entity_item["entity"]

    input_boxes = [bbox]
    raw_image = Image.open(image_path).convert("RGB")
    inputs = sam_processor(raw_image, input_boxes=input_boxes, return_tensors="pt")
    
    pixel_values = inputs.pixel_values.to(device)
    input_boxes = inputs.input_boxes.to(device)
    outputs = sam_model(pixel_values, input_boxes=input_boxes)
    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())

    np_image = np.array(raw_image)
    for ori_mask in masks:
        # color = np.array([123 / 255, 117 / 255, 103 / 255, 1])
        color = np.array([123, 116, 103])
        top1_mask = ori_mask[:, 0, :, :]

        # occluded_mask = torch.where(top1_mask, torch.rand(top1_mask.shape) <= ratio, top1_mask)
        occluded_mask = top1_mask
        
        temp_mask = occluded_mask.clone()
        mask = torch.zeros(temp_mask.shape[-2:], dtype=torch.bool)
        # patch_size = patch_size

        for cnt in range(temp_mask.shape[0]):
            x1, y1, x2, y2 = bbox[cnt]
            patch_size = max(int(min(abs(x2 - x1), abs(y2 - y1)) // 3 * 1), 4)
            cur_step = 0
            next_step = random.randint(0, patch_size // 1)
            gap = int(math.sqrt(patch_size))
            for i in range(0, temp_mask.shape[1], gap):
                for j in range(0, temp_mask.shape[2], gap):
                    if cur_step == next_step:
                        cur_step = 0
                        next_step = random.randint(0, patch_size // 1)
                        if temp_mask[cnt, i, j]:
                            left_i = max(0, i - (patch_size // 2))
                            right_i = min(temp_mask.shape[1] - 1, i + (patch_size // 2))
                            left_j = max(0, j - (patch_size // 2))
                            right_j = min(temp_mask.shape[2] - 1, j + (patch_size // 2))
                            mask[left_i:right_i+1, left_j:right_j+1] = True
                    else:
                        cur_step += 1
                    
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) + ~mask.reshape(h, w, 1) * np_image

        filename = os.path.splitext(os.path.basename(image_path))[0]
        file_path = os.path.join(output_path, f"{filename}-{entity}.jpg")
        cv2.imwrite(file_path, cv2.cvtColor(np.uint8(mask_image), cv2.COLOR_RGB2BGR))
        
        return file_path


whole_image_paths, whole_entities = get_samples()
print(f"Totle Samples: {len(whole_entities)}")


batch = 10000
# for i in range(0, len(whole_entities) // batch + 1):
for i in range(0, len(whole_entities) // batch + 1):
    begin_idx = i * batch
    end_idx = (i + 1) * batch

    print(f"Processing: {begin_idx} ~ {end_idx-1}")
    entities = whole_entities[begin_idx:end_idx]

    images = []
    for entity_item in tqdm(entities):
        dir, filename = os.path.split(entity_item['image_path'])
        dir, split = os.path.split(dir)
        image_path = os.path.join("./datasets/coco/images", split, filename)

        mask_image_path = mask_entity(image_path, entity_item, output_dir)


