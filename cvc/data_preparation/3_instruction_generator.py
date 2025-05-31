from vllm import LLM, SamplingParams

import glob
import json
import tqdm
import os


def get_samples():
    file_dir = "./experiments/entity_high-causality"
    file_name = glob.glob(os.path.join(file_dir, "*"))

    image_path = []
    entity = []
    for _file_name in tqdm.tqdm(file_name):
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



prefix = '''
You are a question constructor. I will provide you with an entity, which could refer to a specific object, a specific type of object, a place, a location or an occupation.
You first need to determine the type of the entity and generate a question according to the type of the given entity.

Here are some constraints you need to follow:
1) If an object is very common, you can simply refer to it as "object".
2) If an object is relatively uncommon, you may need to specify its type in the question.
3) If an object's type is difficult to determine, you can simply refer to it as "object".
4) Should not reveal the entire entity's information in the generated question.

Here are some examples:

Entity: chairs
<begin>
Question: In the given image, there are objects which are heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the objects occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: keyboard
<begin>
Question: In the given image, there is a object heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: sewing machine
<begin>
Question: In the given image, there is a machine that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What kind of machine might the object occluded by a cluster of gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: refrigerator
<begin>
Question: In the given image, there is an appliance that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the appliance occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: pot
<begin>
Question: In the given image, there is an object heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: River Thames
<begin>
Question: In the given image, there is a view of a landmark that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
The view obscured by the gray blocks could be which landmark? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: farm
<begin>
Question: In the given image, there is a view of a place which is heavily occluded by a cluster of gray blocks. Please answer the following question. 
The view obscured by the gray blocks could be which place? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: laptop screen
<begin>
Question: In the given image, there is an object that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: tomato sauce
<begin>
Question: In the given image, there is a condiment heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the condiment occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: fireman
<begin>
Question: In the given image, there is a person engaged in a certain profession, who is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What profession might this partially occluded person be engaged in? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: police officers
<begin>
Question: In the given image, there is a group of person engaged in a certain profession, who are heavily occluded by a cluster of gray blocks. Please answer the following question. 
What profession might the people occluded by the gray blocks be engaged in? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: Queen Elizabeth
<begin>
Question: In the given image, there is a person heavily occluded by a cluster of gray blocks. Please answer the following question. 
Who might the person occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: Barack Obama
<begin>
Question: In the given image, there is a person who is heavily occluded by a cluster of gray blocks. Please answer the following question. 
Who might the person occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: tow truck
<begin>
Question: In the given image, there is a vehicle heavily occluded by a cluster of gray blocks. Please answer the following question. 
What kind of vehicle might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: rider
<begin>
Question: In the given image, there is a view of a place which is heavily occluded by a cluster of gray blocks. Please answer the following question.
The view obscured by the gray blocks could be which place? Please provide your reasoning process and confirm a unique answer.
<end>

Entity: couple
<begin>
Question: In the given image, there is a group of person occluded by a cluster of gray blocks. Please answer the following question. 
What relationship might the people occluded by the gray blocks have? Please provide your reasoning process and confirm a unique answer.

Entity: stick
<begin>
Question: In the given image, there is an object heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: hummingbird
<begin>
Question: In the given image, there is a bird that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What kind of bird might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: chicken
<begin>
Question: In the given image, there is an animal that is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What animal might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: aluminum foil
<begin>
Question: In the given image, there is a material which is heavily occluded by a cluster of gray blocks. Please answer the following question. 
What kind of material might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>

Entity: tooth brush
<begin>
Question: In the given image, there is an object heavily occluded by a cluster of gray blocks. Please answer the following question. 
What might the object occluded by the gray blocks be? Please provide your reasoning process and confirm a unique answer. Let's think step by step.
<end>
'''


ckpt = "meta-llama/Llama-2-7b-chat-hf"

llm = LLM(model=ckpt, tensor_parallel_size=1, enable_prefix_caching=True)

sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=100,
    stop="<end>"
)

_, whole_entity  = get_samples()
whole_entity = whole_entity

print(f"\nTotle Samples: {len(whole_entity)}")

prompts = []
for _entity in whole_entity:
    name = _entity["entity"]
    # prompt = f"[INST] {prefix}\n\nThe given entity is \"{name}\", please provide the question you generated directly. [/INST]"

    prompt = f"{prefix}\nEntity: {name}\n<begin>\nQuestion:"    
    prompts.append(prompt)


outputs = llm.generate(prompts, sampling_params=sampling_params)


output_dir = "./experiments/entity_high-causality_with-generated-instruction"
# output_dir = "./examples"
os.mkdir(output_dir)


for entity_item, output in zip(whole_entity, outputs):
    dir, filename = os.path.split(entity_item['image_path'])
    dir, split = os.path.split(dir)
    image_path = os.path.join("./datasets/coco/images", split, filename)
    entity_item["image_path"] = image_path

    entity_item["query"] = output.outputs[0].text.strip()

    filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
        json.dump(entity_item, f, indent=4)
        f.write("\n")

