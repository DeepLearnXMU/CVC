from vllm import LLM, SamplingParams
from transformers import LlavaProcessor
from PIL import Image
from tqdm import tqdm
import torch
import os

import torch
import json
import glob
import os
from transformers import set_seed
from vllm.sequence import MultiModalData


seed = 0

ckpt = "llava-hf/llava-1.5-7b-hf"
# ckpt = "llava-hf/llava-1.5-13b-hf"


output_dir = f"./experiments/cots"
mask_dir = f"./experiments/masked-images"

set_seed(seed)

# os.mkdir(output_dir)


def get_samples():
    file_dir = "./experiments/entity_high-causality_with-generated-instruction"
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


whole_image_paths, whole_entities = get_samples()
print(f"Totle Samples: {len(whole_entities)}")

llm = LLM(
    model=ckpt,
    image_input_type="pixel_values",
    image_token_id=32000,
    image_input_shape="1,3,336,336",
    image_feature_size=576,
    seed=seed,
    tensor_parallel_size=1,
    swap_space=16
    # enable_prefix_caching=True,
)

sampling_params = SamplingParams(
    n=16,
    temperature=1,
    top_p=0.9,
    max_tokens=384,
)

batch = 2500
# for i in range(10, len(whole_entities) // batch + 1):
for i in range(10, len(whole_entities) // batch + 1):

    begin_idx = i * batch
    end_idx = (i + 1) * batch

    print(f"Processing: {begin_idx} ~ {end_idx-1}")
    entities = whole_entities[begin_idx:end_idx]

    processor = LlavaProcessor.from_pretrained(ckpt)

    prompts = []
    images = []
    for entity_item in tqdm(entities):
        dir, filename = os.path.split(entity_item['image_path'])
        dir, split = os.path.split(dir)
        image_path = os.path.join("./datasets/coco/images", split, filename)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        entity = entity_item["entity"]
        mask_image_path = os.path.join(mask_dir, f"{filename}-{entity}.jpg")
        entity_item["mask_image_path"] = mask_image_path
        
        # query = "In the given image, there is an object which is heavily occluded by a cluster of gray blocks. Please answer the following question.\nWhat might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step."
        query = entity_item["query"]
        
        prompt = "USER: " + "<image>" * 576 + "\n" + query + "\n" + "ASSISTANT:"
        
        prompts.append(prompt)
        
        raw_image = Image.open(mask_image_path)
        pixel_values = processor(prompt, images=raw_image, return_tensors='pt').pixel_values
        
        images.append(pixel_values)
        

    outputs = llm.generate(prompts,
                           multi_modal_data=MultiModalData(type=MultiModalData.Type.IMAGE, data=torch.cat(images, dim=0)), 
                           sampling_params=sampling_params)


    for entity_item, output in zip(entities, outputs):
        entity_item["chain_of_thoughts"] = [response.text.strip() for response in output.outputs]
        
        image_path = entity_item["image_path"]
        filename = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
            json.dump(entity_item, f, indent=4)
            f.write("\n")


