from datasets import Dataset, load_from_disk
import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


SANITY_CHECK_TEMPLATES = {
    "city": {
        "Language": ["People in %s usually speak"],
        "Country": ["%s is in the country of"],
        "Continent": ["%s is in the continent of"]
    }
}


def generate_ravel_dataset(
    tokenizer, 
    n_samples,
    root_path = "./data/ravel/",
    split="train", 
    domains=["city"], 
    domains_excluded_attributes=[["Latitude", "Longitude", "Timezone"]],
    all_templates=None,
    seed=42,
    filter_one_token_entities=False,
    disentangling=False
):
            
    # Seed
    random.seed(seed)
    np.random.seed(seed)
    dataset = []
    
    sample_per_domain = n_samples // len(domains)
    
    for domain, excluded_attributes in zip(domains, domains_excluded_attributes):
        
        if all_templates is None:
            templates = json.load(open(os.path.join(root_path, f"ravel_{domain}_attribute_to_prompts.json"), "r"))
        else:
            templates = all_templates[domain]
            
        templates_split = json.load(open(os.path.join(root_path, f"ravel_{domain}_prompt_to_split.json"), "r"))

        entities = json.load(open(os.path.join(root_path, f"ravel_{domain}_entity_attributes.json"), "r"))
        entities_split = json.load(open(os.path.join(root_path, f"ravel_{domain}_entity_to_split.json"), "r"))
        
        all_attributes = [a for a in list(templates.keys()) if a not in excluded_attributes]

        entities_train = {k: v for k, v in entities.items() if entities_split[k] == "train"}        
        name_train = list(entities_train.keys())
        templates_train = {k: [v for v in vs if templates_split[v] == "train"] for k, vs in templates.items()} if all_templates is None else templates

        entities_test = {k: v for k, v in entities.items() if entities_split[k] != "train"}
        name_test = list(entities_test.keys())
        templates_test = {k: [v for v in vs if templates_split[v] != "train"] for k, vs in templates.items()} if all_templates is None else templates
        
        if split == "train":
            entity_dict, entity_name, template_dict = entities_train, name_train, templates_train
        elif split == "test":
            entity_dict, entity_name, template_dict = entities_test, name_test, templates_test
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        
        if filter_one_token_entities:
            
            filtered_entities = []
            
            for entity in entity_dict.keys():
                
                entity_token_len = len(tokenizer(entity)["input_ids"])
                if entity_token_len > 1:
                    filtered_entities.append(entity)
            
            print(f"Filtered out {len(filtered_entities)} entities")
            entity_dict = {k: v for k, v in entity_dict.items() if k not in filtered_entities}
            entity_name = list(entity_dict.keys())
 
        for _ in tqdm(range(sample_per_domain)):
            
            data = {}
            
            source_entity, base_entity = random.sample(entity_name, 2)
            attribute = random.choice(all_attributes)
            another_attribute = random.choice(all_attributes)
            source_entity_dict, base_entity_dict = entity_dict[source_entity], entity_dict[base_entity]
            source_template, base_template = random.choice(template_dict[another_attribute]), random.choice(template_dict[attribute])
            
            data["input_text"] = base_template % base_entity
            data["entity"] = base_entity
            data["counterfactual_input_text"] = source_template % source_entity
            data["counterfactual_entity"] = source_entity
            data["edit_instruction"] = f"{base_entity} ; {source_entity} - {attribute}"
            data["target"] = base_entity_dict[attribute]
            data["counterfactual_target"] = source_entity_dict[attribute]
            
            data["input_text_with_counterfactual_entity"] = base_template % source_entity
            
            if disentangling:
                disentangled_attributes = [a for a in all_attributes if a != attribute]
                data["disentangled_data"] = {
                    "attributes": disentangled_attributes,
                    "editor_instruction": [],
                    "input_text": [],
                    "counterfactual_input_text": [],
                    "target": [],
                    "counterfactual_target": [],
                    "input_text_with_counterfactual_entity": []
                }
                for a in disentangled_attributes:
                    disentangled_attribute_base_template = random.choice(templates[a])
                    data["disentangled_data"]["editor_instruction"].append(f"{base_entity} ; {source_entity} - {attribute}")
                    data["disentangled_data"]["input_text"].append(disentangled_attribute_base_template % base_entity)
                    data["disentangled_data"]["counterfactual_input_text"].append(source_template % source_entity)
                    data["disentangled_data"]["target"].append(base_entity_dict[a])
                    data["disentangled_data"]["counterfactual_target"].append(source_entity_dict[a])
                    data["disentangled_data"]["input_text_with_counterfactual_entity"].append(disentangled_attribute_base_template % source_entity)
            else:
                data["disentangled_data"] = None
                
            dataset.append(data)
                
    dataset = Dataset.from_list(dataset)
    return dataset

def get_ravel_collate_fn(tokenizer, disentangling=False, contain_entity_position=False, examine_source_output=False):
    
    def tokenize_text_inputs(texts, counterfactual_texts, target_texts, entities=None, counterfactual_entities=None):
        
        input_texts = [text + " " + target for text, target in zip(texts, target_texts)]
        input_texts = [text.replace(" \" ", " \" ") for text in input_texts]
        
        if entities is not None and counterfactual_entities is not None:
            source_entity_position_ids = []
            base_entity_position_ids = []
        
        tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_counterfactual = tokenizer(counterfactual_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_labels = []
        
        for i, (input_ids, input_text), in enumerate(zip(tokenized["input_ids"], texts)):
            input_length = tokenizer(input_text, return_tensors="pt", padding=False)["input_ids"].shape[-1]
            if tokenizer.padding_side == "left":
                input_length += torch.sum(input_ids == tokenizer.pad_token_id)
            
            label = torch.full_like(input_ids, -100)
            label[input_length:] = input_ids[input_length:]
            
            if tokenizer.padding_side == "right":
                label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)
            
            if entities is not None and counterfactual_entities is not None:
                entity_token = entities[i]
                counterfactual_entity_token = counterfactual_entities[i]
                
                base_entity_position_ids.append([tokenizer.decode(ids).strip() for ids in input_ids].index(entity_token))
                source_entity_position_ids.append([tokenizer.decode(ids).strip() for ids in tokenized_counterfactual["input_ids"][i]].index(counterfactual_entity_token))            
        
        tokenized_labels = torch.stack(tokenized_labels)
        
        return_dict = {
            "base_input_ids": tokenized["input_ids"],
            "base_attention_mask": tokenized["attention_mask"],
            "source_input_ids": tokenized_counterfactual["input_ids"],
            "source_attention_mask": tokenized_counterfactual["attention_mask"],
            "labels": tokenized_labels
        }
        
        if entities is not None and counterfactual_entities is not None:
            return_dict["source_entity_position_ids"] = torch.tensor(source_entity_position_ids)
            return_dict["base_entity_position_ids"] = torch.tensor(base_entity_position_ids)
        
        return return_dict
    
    def collate_fn(batch):
        
        prompts, edit_instructions, targets, counterfactual_prompts = [], [], [], []
        if contain_entity_position:
            assert "entity" in batch[0].keys() and "counterfactual_entity" in batch[0].keys()
            entities, counterfactual_entities = [], []
        else:
            entities, counterfactual_entities = None, None
            
        for b in batch:
            
            if not examine_source_output:
                prompts.append(b["input_text"])
            else:
                prompts.append(b["input_text_with_counterfactual_entity"])
                
            edit_instructions.append(b["edit_instruction"])
            counterfactual_prompts.append(b["counterfactual_input_text"])
            targets.append(b["counterfactual_target"])
            
            if contain_entity_position:
                entities.append(b["entity"])
                counterfactual_entities.append(b["counterfactual_entity"])
            
        editor_input_ids = tokenizer(edit_instructions, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        returned_dict = {
            "editor_input_ids": editor_input_ids,
            **tokenize_text_inputs(prompts, counterfactual_prompts, targets, entities=entities, counterfactual_entities=counterfactual_entities),
        }
        
        if disentangling:
            assert batch[0]["disentangled_data"] is not None
            
            disentangled_prompt, disentangled_counterfactual_prompt, disentangled_target, disentangled_example_idxs = [], [], [], []
            disentangled_editor_instructions = []
            
            for i, b in enumerate(batch):
                
                disentangled_prompt.extend(b["disentangled_data"]["input_text"])
                disentangled_counterfactual_prompt.extend(b["disentangled_data"]["counterfactual_input_text"])
                disentangled_target.extend(b["disentangled_data"]["target"])
                disentangled_example_idxs.extend([i] * len(b["disentangled_data"]["input_text"]))
                disentangled_editor_instructions.extend(b["disentangled_data"]["editor_instruction"])
                
            disentangled_editor_input_ids = tokenizer(disentangled_editor_instructions, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            
            disentangled_dict = tokenize_text_inputs(
                disentangled_prompt, 
                disentangled_counterfactual_prompt,
                disentangled_target
            )
            
            disentangled_example_idxs = torch.tensor(disentangled_example_idxs)
            
            returned_dict["disentangled_base_input_ids"] = disentangled_dict["base_input_ids"]
            returned_dict["disentangled_base_attention_mask"] = disentangled_dict["base_attention_mask"]
            returned_dict["disentangled_source_input_ids"] = disentangled_dict["source_input_ids"]
            returned_dict["disentangled_source_attention_mask"] = disentangled_dict["source_attention_mask"]
            returned_dict["disentangled_labels"] = disentangled_dict["labels"]
            returned_dict["disentangled_editor_input_ids"] = disentangled_editor_input_ids
            returned_dict["disentangled_example_idxs"] = disentangled_example_idxs            
        
        return returned_dict
    
    return collate_fn


def filter_dataset(model, tokenizer, dataset, disentangling=False, batch_size=16):
        
    model.eval()
    
    correct_idxs = set()
    
    collate_fn = get_ravel_collate_fn(
        tokenizer, disentangling=disentangling, 
        contain_entity_position=False, examine_source_output=True
    )
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(data_loader)):
            
            prediction = model(
                input_ids=batch["base_input_ids"].to("cuda"),
                attention_mask=batch["base_attention_mask"].to("cuda"),
            )
            
            batch_pred_ids = torch.argmax(prediction["logits"], dim=-1)
            
            if disentangling:
                disentangled_prediction = model(
                    input_ids=batch["disentangled_base_input_ids"].to("cuda"),
                    attention_mask=batch["disentangled_base_attention_mask"].to("cuda"),
                )
                
                disentangled_batch_pred_ids = torch.argmax(disentangled_prediction["logits"], dim=-1)
            
            for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                label_idx = label != -100
                output_idx = torch.zeros_like(label_idx)
                output_idx[:-1] = label_idx[1:]
                
                label = label[label_idx]
                pred_ids = pred_ids[output_idx]
                
                is_correct = (torch.sum(label == pred_ids) == torch.numel(label)).item()
                
                if disentangling:
                    disentangled_label = batch["disentangled_labels"][batch["disentangled_example_idxs"] == i].to("cuda")
                    disentangled_pred_ids = disentangled_batch_pred_ids[batch["disentangled_example_idxs"] == i]
                    
                    for j, (disentangled_label, disentangled_pred) in enumerate(zip(disentangled_label, disentangled_pred_ids)):
                        disentangled_label_idx = disentangled_label != -100
                        disentangled_output_idx = torch.zeros_like(disentangled_label_idx)
                        disentangled_output_idx[:-1] = disentangled_label_idx[1:]
                        
                        disentangled_label = disentangled_label[disentangled_label_idx]
                        disentangled_pred = disentangled_pred[disentangled_output_idx]
                        
                        is_correct_disentangled = (torch.sum(disentangled_label == disentangled_pred) == torch.numel(disentangled_label)).item()
                        
                        is_correct = is_correct and is_correct_disentangled
                
                if is_correct:
                    correct_idxs.add(batch_id * len(batch["labels"]) + i)
    
    print(f"Accuracy: {len(correct_idxs) / len(dataset)}; filtered out {len(dataset) - len(correct_idxs)} examples")
    filtered_dataset = dataset.select(list(correct_idxs))
    return filtered_dataset
            