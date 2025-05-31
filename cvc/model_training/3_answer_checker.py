from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

import glob
import json
import os


def load_entity_with_cots_answers():
    file_dir = "./experiments/cots_with-answer"

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


_, whole_entity_with_cots_answers  = load_entity_with_cots_answers()

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

output_dir = "./experiments/cots_with-answer_checked"
os.mkdir(output_dir)

for entity in tqdm(whole_entity_with_cots_answers):
    image_path = entity['image_path']
    
    predictions = entity["predictions"]
    label = entity["entity"]
    
    embeddings_1 = model.encode(predictions)['dense_vecs']
    embeddings_2 = model.encode(label)['dense_vecs']
    
    similarity = embeddings_1 @ embeddings_2.T
    entity["correctness"] = similarity.tolist()
    
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
        json.dump(entity, f, indent=4)
        f.write("\n")
        