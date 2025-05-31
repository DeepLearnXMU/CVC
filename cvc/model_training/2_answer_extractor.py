from vllm import LLM, SamplingParams
from tqdm import tqdm

import torch
import glob
import json
import os
import re


def load_entity_with_cots():
    file_dir = "./experiments/cots"

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


prefix = '''
You are an answer extractor. I will provide you with a text, which is an analysis of reasoning about what an occluded object is. Please extract the confirmed answers from this reasoning process.


To extract the answer, here are some constraints you need to follow:
1) If the reasoning process does not yield an answer, please determine it as "unknown".
2) If the reasoning process expresses uncertainty or cannot be determined, please determine it as "unknown".
3) Please directly provide the extracted answer, and do not include your analysis reasons.


Here are some examples:

Text: Given the heavily occluded object in the image, it's likely to be a piece of food, possibly a meat, which is covered in a generous amount of toppings, such as onions and cheese. However, without seeing more details, the object's precise type and presence of other ingredients cannot be confirmed.
<begin>
Extracted Answer: meat
<end>

Text: While looking at the image, I noticed that there's a boy in a red beanie and red jacket sitting in the back seat of the bus. The partially visible object that is heavily blocked by gray blocks might be a window. The initial assumption might be that the boy is looking out of the window, which could be obstructed by the gray blocks. However, since the image also shows another person looking out the window, it's more likely that the gray blocks are there to provide privacy or to separate sections of the bus.
<begin>
Extracted Answer: window
<end>

Text: Based on the visual information available in the image, it is impossible to determine the exact nature of the object that is heavily occluded by the gray blocks. However, we can make some educated guesses. One possibility is that the object is a monitor or computer screen, as there is already a computer monitor present in the image. Another possibility is that the object is a lamp or other light source, as there is a lamp present in the image. The gray blocks could also be other random objects, but without more context or visual clues, it is not possible to identify the object with certainty.
<begin>
Extracted Answer: monitor, computer screen, lamp
<end>

Text: Based on the image, the heavily occluded object is in a living room area surrounded by brown furniture and plants. Since there is a couch and a chair present in the scene, it's possible that the object is part of the seating arrangement, such as a bench or an ottoman. The gray blocks could be used as an additional seating or as a side table, which could explain the obscured view of the object. The reasoning process involves identifying the living room elements, such as the couch and the chair, and considering the possibility of an additional seating or side table made up of gray blocks.
<begin>
Extracted Answer: bench, ottoman
<end>

Text: The object heavily obscured by the gray blocks might be a white refrigerator or a microwave placed under a cabinet. The gray blocks are blocking any clear view of the object, making it difficult to determine its exact nature. However, these are two common appliances found in kitchens, and their size and color match the description of the heavily blocked object in the image.
<begin>
Extracted Answer: refrigerator, microwave
<end>

Text: First, we need to identify the edges of the gray blocks. If there is a more noticeable border around the gray blocks, it could be the edge of a cutting board, a plate, or a dish. Now, given the presence of multiple knives in the image, we need to consider what kind of object or material would make it necessary to have multiple knives to work on.\n\nLet's consider the possibility that the gray blocks are a cutting board or a large dish, and the object behind them is a pizza. This could explain the need for multiple knives, as cutting a pizza with various toppings or slices might require different tools.\n\nAlternatively, if the gray blocks are part of a plate or a dish, the object behind them might be sushi or other food items that are traditionally cut using multiple knives.\n\nIn conclusion, we can speculate that the heavily-occluded object could be a pizza or a dish with sushi, given the presence of multiple knives. We cannot provide a definitive answer, as there are many other possibilities. However, these are some plausible scenarios based on the visible information.
<begin>
Extracted Answer: pizza, dish with sushi
<end>

Text: The object obscured by the gray blocks appears to be a baseball. The reason is that there is a person standing in a mound with a baseball, and their outstretched hand seems to be throwing or holding the ball. Considering the context and the activity depicted, it is most likely that the heavily occluded object is a baseball.
<begin>
Extracted Answer: baseball
<end>

Text: Based on the image and the child holding a toy, it is possible that the heavily occluded object might be a soft toy, such as a stuffed animal, a blanket, or a pillow. Soft toys are often made with various materials like fabrics or stuffed with stuffing materials.\n\nNow, let's consider the gray blocks, which seem to be teeth-shaped. Although it is unusual for a toy to be made of teeth-shaped blocks, it could be an unconventional design for a toy. In this case, the object might be a unique or innovative toy. However, without more information, it is not possible to determine the exact object with certainty.
<begin>
Extracted Answer: soft toy
<end>

Text: The object occluded by the gray blocks might be a sink. This is because sinks are often installed in bathrooms alongside toilets, and they are typically positioned under or alongside a mirror. Therefore, it is plausible that the gray blocks are obstructing the view of a sink in the bathroom. However, without more context or a clearer view of the image, it is impossible to confirm this with certainty.",
<begin>
Extracted Answer: sink
<end>

Text: Given the ambiguous nature of the image, it is impossible to definitively determine the object that is being covered by the gray blocks. However, some possibilities could be a skateboard, a backpack, a bicycle, or a mechanical object like a motorcycle. \n\nIn order to narrow down the possibilities, one could focus on the orientation and shape of the object in relation to the gray blocks, but this would only provide an initial clue. Without further information, we cannot conclude a definitive answer.
<begin>
Extracted Answer: skateboard, backpack, bicycle, motorcycle
<end>

Text: Based on the given information, there is an object in the image that is heavily occluded by a cluster of gray blocks. Without more specific information, it's difficult to determine the exact nature of the object. It could be a toilet, sink, or even a tap. The gray blocks may serve as a barrier to block off the area or simply to create an artistic effect. It's important to note that, the word toilet is not mentioned, but the object is mentioned as heavily occluded by grey blocks.
<begin>
Extracted Answer: toilet, sink, tap
<end>

Text: First, I will eliminate the gas stove, as it is not heavily blocked by anything. Now, I will examine the other appliances in the kitchen. I am excluding the microwave, as it is partially covered but not heavily occluded. Next, I am looking at the white dishwasher. However, it is small and not heavily blocked. So, the most likely object heavily blocked by the gray blocks is the dishwasher.
<begin>
Extracted Answer: dishwasher
<end>

Text: The object that is heavily occluded by the gray blocks is the woman's face.
<begin>
Extracted Answer: woman's face
<end>

Text: It is difficult to identify the object fully covered by the gray blocks in the image without a more detailed description. The object's color and shape might be indicators of its nature. However, without more information, it is not possible to provide a unique answer.
<begin>
Extracted Answer: unknown
<end>

Text: The object that is heavily occluded by gray blocks is a cat. This assumption is based on the visible white paws of the cat, which are surrounded by gray blocks. The presence of the cat's paws in the image hints at the cat being partially covered by the blocks, which then becomes an indication that the image is either blurry or that the cat has gray fur, and the gray blocks represent the cat's paws.
<begin>
Extracted Answer: cat's paws
<end>

Text: The heavily occluded object could be an item in the vicinity, like a cup or a laptop. It is surrounded by various other objects such as a bed, bedside table, lamp, and two beds. As the gray blocks surround the object, we can deduce that it is a significant object within the image, and it's possible that it's not properly visible. To determine the exact object, we would need to see more of it without the obstructing gray blocks. It could also be a part of the bed's structure, for example, a bed frame or a base of the bed that is covered by the sheets.
<begin>
Extracted Answer: cup, laptop, bed frame, base of the bed
<end>

Text: First, let's try to identify what might the object be based on what is already visible in the image. The only other visible object that could be an object is the keyboard. So, let's assume that the object is the keyboard.\n\nNext, we'll try to cover the keyboard with the gray blocks. The gray blocks will obscure part of the keyboard while still being visible in the image.\n\nBased on this, we can confirm that the object is the keyboard.
<begin>
Extracted Answer: keyboard
<end>

Text: it's hard to say for sure, but given that the image includes multiple screens, a computer, a keyboard, and a mouse, it's possible that the object is a monitor or a laptop. if it's a monitor, it could be occluded by the gray blocks that are covering a portion of the screen. if it's a laptop, the gray blocks could be covering the screen or the keyboard.
<begin>
Extracted Answer: monitor, laptop
<end>

Text: The object obscured by the gray blocks could be a cabinet or drawer within the kitchen. It is reasonable to think that a cabinet or drawer would be lined up with other cabinets or drawers in a kitchen setup. Given that the image appears to be a kitchen scene with multiple sinks and appliances, it is plausible to assume that the object hidden by the gray blocks could be an item typically found in a kitchen.\n\nFor example, it could be a storage area for utensils, cups, or other kitchen-related items. The step-by-step reasoning process would lead us to considering the various objects that are commonly found in a kitchen setting. While we cannot confidently identify the object from the image, it is reasonable to consider cabinets or drawers as a possibility based on the surrounding elements in the image.
<begin>
Extracted Answer: cabinet, drawer
<end>

Text: The object being heavily occluded by the gray blocks is likely the stove. Stoves often have a range of controls and features which might be responsible for this heavy obscuration.
<begin>
Extracted Answer: stove
<end>

Text: the object occluded by the gray blocks could be any number of things, but based on the context provided, it is most likely that it is a part of a computer mouse. this is because the object is surrounded by the word \"computer mouse\" and the term \"small graphic\" suggests that the object may have a mouse-related function or purpose.
<begin>
Extracted Answer: computer mouse
<end>

Text: The object hidden by the gray blocks could be a baseball bat, as it is positioned vertically and takes up a significant part of the image. It is important to confirm this answer by noticing other unique features of the scene or any additional information provided. However, based on the given context, it is plausible that the object hidden by the gray blocks is a baseball bat.
<begin>
Extracted Answer: baseball bat
<end>

Text: The image shows a brown dog lying on a bed, with gray blocks partially covering the bed. Considering the dog is the most prominent object in the image, it is plausible that the object partially blocked by the gray blocks is the dog. However, without more information or context, this answer is speculative, and it's difficult to confirm the exact nature of the hidden object. It could be anything from the dog's tail to another pet, a piece of furniture or even a person's arm.
<begin>
Extracted Answer: dog, dog's tail, another pet, piece of furniture, person's arm
<end>

Text: Based on the image, the object that might be occluded by the gray blocks is the toilet bowl itself. The gray blocks are placed in such a way that they look like they are covering the toilet bowl, suggesting that it is the focus of the image. Additionally, the rest of the toilet is white, and there are no other objects that are heavily occluded by gray blocks.
<begin>
Extracted Answer: toilet bowl
<end>

Text: The heavily-occluded object in the image is a flat-screen TV mounted on the wall.
<begin>
Extracted Answer: flat-screen TV
<end>

Text: The object obstructed by the gray blocks is most likely a cell phone. This assumption is based on the fact that the man is using a cell phone while smiling. It is common for people to use their phones to communicate or access information while maintaining a positive demeanor. In this case, the man's smile and the image of the cell phone all contribute to the likelihood that the object obstructed by the gray blocks is indeed a cell phone.
<begin>
Extracted Answer: cell phone
<end>

Text: The heavily occluded object in the image could be a giraffe that is drinking water from a watering hole. Based on the given context, the primary focus of the image seems to be the giraffe, which would be consistent with the watering hole. However, the presence of gray blocks covering the image can create confusion. It's important to note that giraffes are generally found in the wild, rather than in a crowded place, and a watering hole usually has few or no obstructions. Considering these factors, the most likely explanation is that the gray blocks represent a man-made environment, such as a zoo or a safari park, where a giraffe is seen drinking water. The objects might be the gates or structures that are preventing the object from fully visible, not the watering hole itself.
<begin>
Extracted Answer: giraffe
<end>

Text: One possible answer is that the object is a person, possibly a racer or a spectator, since the person is wearing a red and white uniform. The gray blocks could represent the barriers or guard rails that are often present in racing environments to protect the crowd and racers. The heavily blocked area may indicate that the person is sitting in a dangerous spot with minimal protection. This unique answer provides context to the situation presented in the image, as it relates to a racing event where the racers are attempting to outpace each other and the gray blocks are barriers designed to maintain safety during the race.
<begin>
Extracted Answer: racer, spectator
<end>
'''

ckpt = "meta-llama/Llama-2-7b-chat-hf"

llm = LLM(model=ckpt, tensor_parallel_size=4, enable_prefix_caching=True)

sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=100,
    stop="<end>"
)

_, whole_entity_with_cots  = load_entity_with_cots()
print(f"\nTotle Samples: {len(whole_entity_with_cots)}")
# whole_entity_with_cots = whole_entity_with_cots[:100]

output_dir = "./experiments/cots_with-answer"
# output_dir = "./examples"
os.mkdir(output_dir)

batch = 1000
for i in range(0, len(whole_entity_with_cots) // batch + 1):
    
    begin_idx = i * batch
    end_idx = (i + 1) * batch

    print(f"\nProcessing: {begin_idx} ~ {end_idx-1}")
    entity_with_cots = whole_entity_with_cots[begin_idx:end_idx]
    
    cot_cnt = []
    prompts = []
    for _entity_with_cots in entity_with_cots:
        cots = _entity_with_cots["chain_of_thoughts"]
        label = _entity_with_cots["entity"]
        for cot in cots:
            # prompt = f"[INST] {prefix}\n\nPlease follow the extracted format.\nHere is the given text: \"{cot}\"\nHere is the given label: \"{label}\" [/INST]"
            prompt = f"{prefix}\nText: {cot}\n<begin>\n"
            prompts.append(prompt)

        cot_cnt.append(len(cots))

    cot_cnt_cumsum = torch.tensor(cot_cnt).cumsum(dim=0).tolist()

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for idx, entity in enumerate(entity_with_cots):
        dir, filename = os.path.split(entity['image_path'])
        dir, split = os.path.split(dir)
        image_path = os.path.join("./datasets/coco/images", split, filename)
        entity["image_path"] = image_path

        if idx == 0:
            output = outputs[:cot_cnt_cumsum[idx]]
        else:
            output = outputs[cot_cnt_cumsum[idx-1]:cot_cnt_cumsum[idx]]

        output_text = []
        for _output in output:
            output_text.append(_output.outputs[0].text.strip())
        predictions = []
        results = []
        for _output_text in output_text:
            prediction = ""
            for line in _output_text.split('\n'):
                answer_match = re.search("Extracted Answer:", line)
                if answer_match:
                    prediction = line[answer_match.end():].strip()
                    break
            predictions.append(prediction)
            
        entity["predictions"] = predictions

        filename = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
            json.dump(entity, f, indent=4)
            f.write("\n")

