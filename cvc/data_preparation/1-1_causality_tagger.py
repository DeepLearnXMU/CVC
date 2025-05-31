from transformers import AutoModelForMaskedLM, AutoTokenizer
from stanfordcorenlp import StanfordCoreNLP

import torch
import tqdm
import json
import glob
import os


def get_samples():
    file_dir = "./experiments/entity"
    file_name = glob.glob(os.path.join(file_dir, "*"))

    image_path = []
    text_with_entities = []
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
                _text_with_entities = json.loads(obj)
            
                image_path.append(_text_with_entities["image_path"])
                text_with_entities.append(_text_with_entities)
    
    return image_path, text_with_entities


def find_all_subsequences(main_sequence, sub_sequence):
    start = 0
    while True:
        try:
            start = main_sequence.index(sub_sequence[0], start)
        except ValueError:
            break
        if main_sequence[start:start+len(sub_sequence)] == sub_sequence:
            yield (start, start+len(sub_sequence))
        start += 1


def get_mask_text(text_tokens, entity_item, tokenizer, type):
    start_idx = entity_item["start_idx"]
    end_idx = entity_item["end_idx"]
    
    assert type in ["full_form", "label"]
    if type == "full_form":
        entity = entity_item["full_form"]
    elif type == "label":
        entity = entity_item["label"]
        
    if start_idx == 0:
        _temp_start_text = ""
        _temp_end_text = entity
    else:
        _temp_start_text = " ".join(text_tokens[:start_idx])
        _temp_end_text = " ".join([_temp_start_text, entity])

    _temp_start_ids = tokenizer.encode(_temp_start_text, add_special_tokens=False)
    _temp_end_ids = tokenizer.encode(_temp_end_text, add_special_tokens=False)
    entity_ids = _temp_end_ids[len(_temp_start_ids):]
    
    mask_text = " ".join(text_tokens[:start_idx] + [tokenizer.mask_token] * len(entity_ids) + text_tokens[end_idx:])
    mask_text = tokenizer.decode(tokenizer.encode(mask_text, add_special_tokens=False))
        
    return mask_text, entity_ids



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "FacebookAI/roberta-large"
model = AutoModelForMaskedLM.from_pretrained(model_path, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()



whole_image_path, whole_text_with_entities = get_samples()

bsz = 600000
i = 3
begin_idx = i * bsz
end_idx = (i + 1) * bsz
whole_image_path = whole_image_path[begin_idx:end_idx]
whole_text_with_entities = whole_text_with_entities[begin_idx:end_idx]
print(f"Processing: {begin_idx} ~ {end_idx-1}")

# Download from https://stanfordnlp.github.io/CoreNLP/download.html
with StanfordCoreNLP("./stanford-corenlp-4.5.6", port=7891+i, lang="en") as nlp:
    
    output_dir = "./experiments/entity_with-causality"
    
    # os.mkdir(output_dir)

    for image_path, text_with_entities in tqdm.tqdm(zip(whole_image_path, whole_text_with_entities), total=len(whole_image_path)):
        mask_text = []
        mask_gt_ids = []
        mask_gt_text = []
        
        text = text_with_entities["text"]
        entities = text_with_entities["entities"]
        text_tokens = nlp.word_tokenize(text)

        entity_items = []
        for entity in entities:
            if len(entity) != 2:
                continue
            _full_form, _label = entity
            if _full_form == "" or _label == "":
                continue
            
            full_form_tokens = nlp.word_tokenize(_full_form)
            if len(full_form_tokens) == 0:
                continue
            
            positions = list(find_all_subsequences(text_tokens, full_form_tokens))
            for start_idx, end_idx in positions:
                entity_items.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "full_form": _full_form,
                    "label": _label
                })

        for entity_item in entity_items:
            start_idx = entity_item["start_idx"]
            end_idx = entity_item["end_idx"]
            _full_form = entity_item["full_form"]
            _label = entity_item["label"]
            
            _full_form_mask_text, _full_form_entity_ids = get_mask_text(text_tokens, entity_item, tokenizer, type="full_form")
            _label_mask_text, _label_entity_ids = get_mask_text(text_tokens, entity_item, tokenizer, type="label")
            
            mask_text.append(_full_form_mask_text)
            mask_text.append(_label_mask_text)
            
            mask_gt_ids.append(_full_form_entity_ids)
            mask_gt_ids.append(_label_entity_ids)

            mask_gt_text.append(_full_form)
            mask_gt_text.append(_label)
            
        if len(mask_text) == 0:
            continue    
            
        inputs = tokenizer(mask_text, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits

        entity_data = []
        for bs in range(0, logits.shape[0], 2):
            
            def get_ppl(_logits, _input_ids, _mask_gt_ids, _tokenizer):
                _mask_logits = _logits[_input_ids == _tokenizer.mask_token_id]
                _mask_prob = torch.softmax(_mask_logits, dim=-1)
                _label_prob = torch.gather(_mask_prob, dim=-1, index=torch.LongTensor(_mask_gt_ids)[:, None].to(device))
                _ppl = torch.exp(-1 * _label_prob.log().sum() / _label_prob.shape[0])

                return _ppl
            
            full_form_entity_ppl = get_ppl(logits[bs], inputs["input_ids"][bs], mask_gt_ids[bs], tokenizer)
            label_entity_ppl = get_ppl(logits[bs+1], inputs["input_ids"][bs+1], mask_gt_ids[bs+1], tokenizer)
            item = {}
            item["mask_text"] = [mask_text[bs], mask_text[bs+1]]
            item["entity"] = [mask_gt_text[bs], mask_gt_text[bs+1]]
            item["label_tokens"] = [tokenizer.convert_ids_to_tokens(mask_gt_ids[bs]), tokenizer.convert_ids_to_tokens(mask_gt_ids[bs+1])]
            item["ppl"] = [float(full_form_entity_ppl), float(label_entity_ppl)]
            entity_data.append(item)

        output_data = {}
        output_data["image_path"] = image_path
        output_data["text"] = text
        output_data["entity_data"] = sorted(entity_data, key=lambda item : item["ppl"])
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, f"{filename}.json"), mode="a") as f:
            json.dump(output_data, f, indent=4)
            f.write("\n")

