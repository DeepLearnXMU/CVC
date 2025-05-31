from vllm import LLM, SamplingParams

import json
import tqdm
import os
import re


def get_samples():
    pairs = {}
    
    with open("./coco_train.jsonlist", mode="r") as f:
        coco_dataset = json.load(f)
    for sample in coco_dataset:
        image_path = sample["image"]
        text = sample["text_input"]
        
        if image_path not in pairs:
            pairs[image_path] = [text]
        else:
            pairs[image_path].append(text)


    return image_path, text

    
prefix = '''
You are an entity extractor that helps me extract entities from text.
I will provide you with a piece of text, which is a description of an image. Since I need to get the textual descriptions of the entities that appear in the image, I need you to extract entities from the given text as accurately as possible.


Here are some definitions of entity:
1) Entities are concrete objects that can be recognized in the corresponding image (such as objects, people, places, products).
2) The entities to be extracted do not need to be named entities; as long as they are concrete objects, they can be extracted.
3) Entities can appear in text in various forms, including single word, noun phrases.
4) For abstract concept nouns or noun phrases (such as emotions, states, qualities, sensations, actions), they are not considered as entities.


Here are some constraints you need to follow:
1) The form of entities does not consider any format other than single word and noun phrases.
2) If a noun phrase is not an entity, then the nouns within that phrase cannot be considered as entities either.
3) To extract an entity, you first need to identify the complete expression and then remove any modifiers. The text format is as follows: "1. {entity w/ modifiers} -> {entity w/o modifiers}".
4) Please do not extract modifiers indicating quantity.


Here are some examples:

Text: People sunbathing and sitting under umbrellas at a city beach.
Extracted entities:
<begin>
1. people -> people
2. umbrellas -> umbrellas
3. city beach -> city beach
<end>

Text: The image is a collage of various kitchen items, including a clock, pots, pans, a refrigerator, and a chalkboard.
Extracted entities:
<begin>
1. clock -> clock
2. pots -> pots
3. pans -> pans
4. refrigerator -> refrigerator
5. chalkboard -> chalkboard
<end>

Text: A male tennis player hits the ball on a grass court.
Extracted entities:
<begin>
1. male tennis player -> tennis player
2. ball -> ball
3. grass court -> grass court
<end>

Text: A group of people walks down a train platform, with a yellow train stopped nearby.
Extracted entities:
<begin>
1. people -> people
2. train platform -> train platform
3. yellow train -> train
<end>

Text: A motorbike parked on a roadside close to some bush.
Extracted entities:
<begin>
1. motorbike -> motorbike
2. bush -> bush
<end>

Text: Several suitcases, a pair of shoes and a hand bag in a hallway.
Extracted entities:
<begin>
1. suitcases -> suitcases
2. shoes -> shoes
3. hand bag -> hand bag
4. hallway -> hallway
<end>

Text: A freshly baked pepperoni and olive pizza, still on the white plate it was cooked on, sits on a wooden table.
Extracted entities:
<begin>
1. freshly baked pepperoni and olive pizza -> pepperoni and olive pizza
2. white plate -> plate
3. wooden table -> table
<end>

Text: A clock mounted on a brick wall with Roman numerals, above a company's name and date of establishment.
Extracted entities:
<begin>
1. clock -> clock
2. brick wall -> brick wall
3. company's name -> company's name
<end>

Text: A collage of four street signs for Danforth Avenue, mounted on different surfaces and locations, showcasing the sign's visibility and importance in the city.
Extracted entities:
<begin>
1. street signs -> street signs
2. Danforth Avenue -> Danforth Avenue
<end>

Text: A TV displaying a reporter giving the daily news.
Extracted entities:
<begin>
1. TV -> TV
2. reporter -> reporter
<end>

Text: A view from a car window at an intersection on the street.
Extracted entities:
<begin>
1. car window -> car window
2. street -> street
<end>

Text: A container of soup next to the container lid and a plastic spoon.
Extracted entities:
<begin>
1. soup -> soup
2. container lid -> container lid 
3. plastic spoon -> spoon
<end>

Text: A bowl of delicious stir-fried vegetables and tofu, served on a checkered tablecloth.
Extracted entities:
<begin>
1. delicious stir-fried vegetables -> vegetables
2. tofu -> tofu
3. checkered tablecloth -> tablecloth
<end>

Text: Several suitcases, a pair of shoes and a hand bag in a hallway.
Extracted entities:
<begin>
1. suitcases -> suitcases
2. shoes -> shoes
3. hand bag -> hand bag
4. hallway -> hallway
<end>

Text: an air plane flying thru the air with the sky in the abckground
Extracted entities:
<begin>
1. air plane -> air plane
2. sky -> sky
<end>

Text: Two laptop computers open and turned on in a dining room.
Extracted entities:
<begin>
1. laptop computers -> laptop computers
2. dining room -> dining room
<end>

Text: Two adults making bizarre faces for a photo shot while drinking beer.
Extracted entities:
<begin>
1. adults -> adults
2. beer -> beer
<end>

Text: A wooden table topped with fresh produce, including corn, broccoli, cherries, and tomatoes.
Extracted entities:
<begin>
1. wooden table -> table
2. fresh produce -> produce
3. corn -> corn
4. broccoli -> broccoli
5. cherries -> cherries
6. tomatoes -> tomatoes
<end>

Text: A hot dog with chili and a side of potato chips.
Extracted entities:
<begin>
1. hot dog -> hot dog
2. chili -> chili
3. potato chips -> potato chips
<end>

Text: A snowboarder in a black jacket and hat sits in the snow with their snowboard, possibly taking a break or contemplating their next move.
Extracted entities:
<begin>
1. snowboarder -> snowboarder
2. black jacket -> jacket
3. hat -> hat
4. snow -> snow
5. snowboard -> snowboard
<end>

Text: A snowboarder takes a break on the snowy slopes, surrounded by a beautiful winter landscape.
Extracted entities:
<begin>
1. snowboarder -> snowboarder
2. snowy slopes -> snowy slopes
<end>

Text: A variety of tools needed for a project sitting on a table.
Extracted entities:
<begin>
1. tools -> tools
2. table -> table
<end>

Text: An assortment of neckties, all with various colors and patterns.
Extracted entities:
<begin>
1. neckties -> neckties
<end>

Text: An old, stained elaborate roman numerical clock on a clear day
Extracted entities:
<begin>
1. old, stained elaborate roman numerical clock -> clock
<end>

Text: Two tiny stuffed decorations and a cereal box and storage container sit atop a refrigerator, close to a hand-scrawled message on the door.
Extracted entities:
<begin>
1. tiny stuffed decorations -> decorations
2. cereal box -> box
3. storage container -> storage container
4. refrigerator -> refrigerator
5. hand-scrawled message -> message
6. door -> door
<end>

Text: A shiny, red apple on top of other apples.
Extracted entities:
<begin>
1. shiny, red apple -> apple
2. other apples -> apples
<end>

Text: a small child wearing a tie and holding a suitcase with other people in the background
Extracted entities:
<begin>
1. small child -> child
2. tie -> tie
3. suitcase -> suitcase
4. other people -> people
<end>

Text: The kitchen is clean and ready for us to see.
Extracted entities:
<begin>
1. kitchen -> kitchen
<end>

Text: Brick sidewalk with parked motorcycle, firehose in wall and street sign with arrow and foreign words.
Extracted entities:
<begin>
1. brick sidewalk -> brick sidewalk
2. parked motorcycle -> motorcycle
3. firehose -> firehose
4. wall -> wall
5. street sign -> street sign
6. arrow -> arrow
7. foreign words -> foreign words
<end>

Text: On the table there is a plate with beef and a bowl of soup, a plate with a potato, a plate of broccoli, a basket of bread, and a drink.
Extracted entities:
<begin>
1. table -> table
2. plate -> plate
3. beef -> beef
4. soup -> soup
5. potato -> potato
6. broccoli -> broccoli
7. bread -> bread
8. drink -> drink
<end>

Text: A thick evergreen forest marks the boundary of a dark expanse of water, on which rests a long boat with packages at the rear and people to the fore, several holding long oars.
Extracted entities:
<begin>
1. thick evergreen forest -> forest
2. water -> water
3. long boat -> boat
4. packages -> packages
5. people -> people
6. long oars -> oars
<end>

Text: The man skis through the snowy field with another skier in the distance.
Extracted entities:
<begin>
1. man -> man
2. snowy field -> snowy field
3. another skier -> skier
<end>


Please follow the extracted format.
Here is the given text: '''


ckpt = "meta-llama/Llama-2-7b-chat-hf"

llm = LLM(model=ckpt, tensor_parallel_size=1, enable_prefix_caching=True)


whole_image_path, whole_text = get_samples()

output_dir = "./experiments/entity"
# os.mkdir(output_dir)

batch = 20000
# for i in range(10, len(whole_image_path) // batch + 1):
for i in range(90, len(whole_image_path) // batch + 1):

    begin_idx = i * batch
    end_idx = (i + 1) * batch

    print(f"Processing: {begin_idx} ~ {end_idx-1}")
    image_path = whole_image_path[begin_idx:end_idx]
    text = whole_text[begin_idx:end_idx]

    prompts = []
    for _text in text:
        prompt = f"[INST] {prefix}\"{_text}\" [/INST]"
        prompts.append(prompt)


    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=100,
        stop="<end>",
        # stop_token_ids=[2]
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for _image_path, _text, output in zip(image_path, text, outputs):
        output_text = output.outputs[0].text.strip()

        entity= []
        for line in output_text.split('\n'):
            match = re.search("[1-9][0-9]?\.", line)
            if match:
                _entity = [item.strip() for item in line[match.end():].split("->")]
                entity.append(_entity)

        output_data = {}
        output_data["image_path"] = _image_path
        output_data["text"] = _text
        output_data["entities"] = entity

        filename = os.path.splitext(os.path.basename(_image_path))[0]
        with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
            json.dump(output_data, f, indent=4)
            f.write("\n")

