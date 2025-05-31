import glob
import json
import os

import numpy as np
import random

random.seed(0)

from tqdm import tqdm


def load_entity_with_cots_answers():

    file_dirs = [
        "./experiments/cots_with-answer_checked",
        ]

    
    image_path = []
    entity = []
    
    for file_dir in file_dirs:
        file_name = glob.glob(os.path.join(file_dir, "*"))
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

image_paths, entities = load_entity_with_cots_answers()


my_data_cots = []
my_data_answers = []
for entity in entities:
    mask_image_path = entity["mask_image_path"]

    if "query" in entity:
        query = entity["query"]
    else:
        query = "In the given image, there is an object which is heavily occluded by a cluster of gray blocks. Please answer the following question.\nWhat might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step."
    
    
    correctness = np.array(entity["correctness"])
    flag = correctness > 0.85
    acc_cnt = flag.sum()

    id_name = os.path.basename(os.path.splitext(os.path.basename(mask_image_path))[0])
    if acc_cnt >= 1 and acc_cnt <= 4:
        cots = entity["chain_of_thoughts"]
        
        acc_cots = np.array(cots)[flag].tolist()
        
        for idx, acc_cot in enumerate(acc_cots):
            my_data_cots.append(
                {
                    "id": f"{id_name}-cot-{idx}",
                    "image": mask_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{query} Please provide your reasoning process and confirm a unique answer. Let's think step by step."
                        },
                        {
                            "from": "gpt",
                            "value": acc_cot
                        }
                    ]
                }
            )
            
    if acc_cnt >= 1:
        answer_set = set()
        
        predictions = entity["predictions"]
        
        acc_predictions = np.array(predictions)[flag].tolist()
        for idx, acc_prediction in enumerate(acc_predictions):
            if acc_prediction in answer_set:
                continue
            else:
                answer_set.add(acc_prediction)

            my_data_answers.append(
                {
                    "id": f"{id_name}-answer-{idx}",
                    "image": mask_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{query} Please confirm a unique answer."
                        },
                        {
                            "from": "gpt",
                            "value": acc_prediction
                        }
                    ]
                }
            )
            
with open("./llava_v1_5_mix665k.json") as f:
    llava_data = json.load(f)

with open("./experiments/my-data.json", mode="w") as f:
    json.dump(llava_data + my_data_cots + my_data_answers, f, indent=4)

