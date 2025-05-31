from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from tqdm import tqdm
from PIL import Image

import pandas as pd
import numpy as np

import glob
import json
import os


min_ppl = 1 # prob=1
max_ppl = 3.33 # prob=0.3
detect_threshold = 0.6
sample_seed = 0
output_dir = f"/path/to/experiment/entity_high-causality"
# output_dir = f"./examples"

# os.mkdir(output_dir)


def init_glip():
    config_file = "/path/to/model_cards/glip/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/path/to/model_cards/glip/MODEL/glip_large_model.pth"

    # update the config options with the config file
    # manual override some options
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip = GLIPDemo(
        cfg,
        min_image_size=800,
        show_mask_heatmaps=False
    )

    return glip


def load_image(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    
    return image


def load_entity():
    file_dir = "/path/to/experiment/entity_with-causality"
    file_name = glob.glob(os.path.join(file_dir, "*"))

    data = []
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
                try:
                    _entity = json.loads(obj)
                except:
                    print(_file_name)
            
                data.append(_entity)


    entity_data = []
    for _data in tqdm(data):
        for _entity_data in _data['entity_data']:
            entity_data.append(
                {
                    'image_path': _data['image_path'],
                    'text': _data['text'],
                    'entity': _entity_data['entity'][1],
                    'ppl': _entity_data['ppl'][1],
                }
            )
    
    return pd.DataFrame(entity_data)


glip = init_glip()

entity_df = load_entity()

reserved_entity_df = entity_df[(entity_df['ppl'] > min_ppl) & (entity_df['ppl'] < max_ppl)].sort_values(by='ppl')

batch = 150000
i = 1
begin_idx = i * batch
end_idx = (i + 1) * batch

print(f"\nProcessing: {begin_idx} ~ {end_idx-1}")

reserved_entity_df = reserved_entity_df.iloc[begin_idx:end_idx]

for index, entity in tqdm(reserved_entity_df.iterrows(), total=len(reserved_entity_df)):
    item = entity.to_dict()

    dir, filename = os.path.split(item['image_path'])
    dir, split = os.path.split(dir)
    image_path = os.path.join("./coco/images", split, filename)
    item["image_path"] = image_path
    
    image = load_image(item["image_path"])
    
    predictions = glip.compute_prediction(image, item["entity"])
    top_predictions = glip._post_process(predictions, threshold=detect_threshold)
    
    if top_predictions.bbox.shape[0] != 0:
        item["bbox"] = top_predictions.bbox.cpu().numpy().tolist()
        item["score"] = top_predictions.get_field("scores").cpu().numpy().tolist()

        filename = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
            json.dump(item, f, indent=4)
            f.write("\n")

