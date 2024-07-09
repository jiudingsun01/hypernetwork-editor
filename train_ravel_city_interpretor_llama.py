max_grad_clip = 4.0
lr = 1e-4

import torch

torch.manual_seed(42)
# torch.set_default_device("cuda")

import torch
from torch import compile
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import time
import sys

# sys.path.append("..")

import wandb

wandb.init(
    project="hypernetworks-interp",
    config={"targetmodel": "llama3-8b", "editormodel": "llama3-8b", "dataset": "ravel"},
)


from transformers import PreTrainedTokenizerFast, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1")
tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-HF")
tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

import random
import numpy as np
import json
from tqdm import tqdm
from datasets import Dataset

def generate_ravel_dataset(n_samples, split="train", domains=["city"], domain_excluded_attributes=[["Latitude", "Longitude", "Timezone"]], filtering_dict_paths=[None],  seed=42):
            
    # Seed
    random.seed(seed)
    np.random.seed(seed)
    dataset = []
    
    sample_per_domain = n_samples // len(domains)
    
    for domain, excluded_attributes, filtering_dict_path in zip(domains, domain_excluded_attributes, filtering_dict_paths):
                        
        templates = json.load(open(os.path.join("./data/ravel/", f"ravel_{domain}_attribute_to_prompts.json"), "r"))
        entities = json.load(open(os.path.join("./data/ravel/", f"ravel_{domain}_entity_attributes.json"), "r"))
        entities_split = json.load(open(os.path.join("./data/ravel/", f"ravel_{domain}_entity_to_split.json"), "r"))
        templates_split = json.load(open(os.path.join("./data/ravel/", f"ravel_{domain}_prompt_to_split.json"), "r"))
        
        all_attributes = [a for a in list(templates.keys()) if a not in excluded_attributes]

        templates_train = {k: [v for v in vs if templates_split[v] == "train"] for k, vs in templates.items()}
        templates_train_idxs = {k: [templates[k].index(v) for v in vs if templates_split[v] == "train"] for k, vs in templates.items()}
        
        templates_test = {k: [v for v in vs if templates_split[v] != "train"] for k, vs in templates.items()}
        templates_test_idxs = {k: [templates[k].index(v) for v in vs if templates_split[v] != "train"] for k, vs in templates.items()}

        entities_train = {k: v for k, v in entities.items() if entities_split[k] == "train"}
        name_train = list(entities_train.keys())

        entities_test = {k: v for k, v in entities.items() if entities_split[k] != "train"}
        name_test = list(entities_test.keys())
        
        if split == "train":
            entity_dict, entity_name, prompt_dict, prompt_idxs_dict = entities_train, name_train, templates_train, templates_train_idxs
        elif split == "test":
            entity_dict, entity_name, prompt_dict, prompt_idxs_dict = entities_test, name_test, templates_test, templates_test_idxs
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        if filtering_dict_path is not None:
            filtering_dict = json.load(open(filtering_dict_path, "r"))
            filtered_key = []
            
            for entity in filtering_dict.keys():
                model_knows = True
                for attribute in all_attributes:                    
                    split_template_idx = [list(filtering_dict[entity][attribute].values())[i] for i in prompt_idxs_dict[attribute]]
                    if True not in split_template_idx:
                        model_knows = False
                        break
                if not model_knows:
                    filtered_key.append(entity)
            print(f"Filtering out {len(filtered_key)} out of {len(filtering_dict)} entities that the model does not know!")
            filtering_dict = {k: v for k, v in filtering_dict.items() if k not in filtered_key}
        else:
            filtering_dict = None
        
        for _ in tqdm(range(sample_per_domain)):
            
            data = {}
            
            if filtering_dict is None:
                source_entity, base_entity = random.sample(entity_name, 2)
                attribute = random.choice(all_attributes)
                frozen_attributes = [k for k in all_attributes if k != attribute]
                source_entity_dict, base_entity_dict = entity_dict[source_entity], entity_dict[base_entity]
                source_template, base_template = random.choice(prompt_dict[attribute]), random.choice(prompt_dict[attribute])
            else:
                source_entity, base_entity = random.sample([k for k in entity_name if k in filtering_dict.keys()], 2)
                attribute = random.choice(all_attributes)
                frozen_attributes = [k for k in all_attributes if k != attribute]
                source_entity_dict, base_entity_dict = entity_dict[source_entity], entity_dict[base_entity]
                source_template_idxs = [i for i in range(len(filtering_dict[source_entity][attribute])) if filtering_dict[source_entity][attribute][str(i)] == True]
                source_template_idxs = [prompt_idxs_dict[attribute].index(i) for i in source_template_idxs if i in prompt_idxs_dict[attribute]]
                source_template = random.choice([prompt_dict[attribute][i] for i in source_template_idxs])
                base_template_idxs = [i for i in range(len(filtering_dict[base_entity][attribute])) if filtering_dict[base_entity][attribute][str(i)] == True]
                base_template_idxs = [prompt_idxs_dict[attribute].index(i) for i in base_template_idxs if i in prompt_idxs_dict[attribute]]
                base_template = random.choice([prompt_dict[attribute][i] for i in base_template_idxs])
                
            data["input_text"] = base_template % base_entity
            data["counterfactual_input_text"] = source_template % source_entity
            data["edit_instruction"] = f"{base_entity} ; {source_entity} - {attribute}"
            data["target"] = base_entity_dict[attribute]
            data["counterfactual_target"] = source_entity_dict[attribute]
            
            data["unaffected_attributes"] = []
            
            for frozen_attribute in frozen_attributes:
                
                if filtering_dict is None:
                    source_attribute_template, base_attribute_template = random.choice(prompt_dict[frozen_attribute]), random.choice(prompt_dict[frozen_attribute])
                else:
                    source_attribute_idxs = [i for i in range(len(filtering_dict[source_entity][frozen_attribute])) if filtering_dict[source_entity][frozen_attribute][str(i)] == True]
                    source_attribute_idxs = [prompt_idxs_dict[frozen_attribute].index(i) for i in source_attribute_idxs if i in prompt_idxs_dict[frozen_attribute]]
                    source_attribute_template = random.choice([prompt_dict[frozen_attribute][i] for i in source_attribute_idxs])
                    base_attribute_idxs = [i for i in range(len(filtering_dict[base_entity][frozen_attribute])) if filtering_dict[base_entity][frozen_attribute][str(i)] == True]
                    base_attribute_idxs = [prompt_idxs_dict[frozen_attribute].index(i) for i in base_attribute_idxs if i in prompt_idxs_dict[frozen_attribute]]
                    base_attribute_template = random.choice([prompt_dict[frozen_attribute][i] for i in base_attribute_idxs])
                    
                base_prompt = base_attribute_template % base_entity
                counterfactual_prompt = source_attribute_template % source_entity
                
                target = base_entity_dict[frozen_attribute]
                counterfactual_target = source_entity_dict[frozen_attribute]
                
                data["unaffected_attributes"].append(
                    {
                        "input_text": base_prompt,
                        "counterfactual_input_text": counterfactual_prompt,
                        "edit_instruction": f"{base_entity} ; {source_entity} - {attribute}",
                        "target": target,
                        "counterfactual_target": counterfactual_target,
                    }
                )
                
            dataset.append(data)
                
    dataset = Dataset.from_list(dataset)
    return dataset

city_train_set = generate_ravel_dataset(20480, split="train", filtering_dict_paths=["./notebooks/ravel_llama-3-8b_city_prompt_to_output_statistics.json"])
city_test_set =  generate_ravel_dataset(1024, split="test", filtering_dict_paths=["./notebooks/ravel_llama-3-8b_city_prompt_to_output_statistics.json"])

def ravel_collate_fn(batch):
    
    def tokenize_text_inputs(texts, counterfactual_texts, target_texts):
        
        input_texts = [text + " " + target for text, target in zip(texts, target_texts)]
        input_texts = [text.replace(" \" ", " \" ") for text in input_texts]
        
        tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_counterfactual = tokenizer(counterfactual_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_labels = []
        
        for input_ids, input_text in zip(tokenized["input_ids"], texts):
            input_length = tokenizer(input_text, return_tensors="pt", padding=False)["input_ids"].shape[-1]
            label = torch.full_like(input_ids, -100)
            label[input_length:] = input_ids[input_length:]
            label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)
        
        tokenized_labels = torch.stack(tokenized_labels)
        return {
            "base_input_ids": tokenized["input_ids"],
            "base_attention_mask": tokenized["attention_mask"],
            "source_input_ids": tokenized_counterfactual["input_ids"],
            "source_attention_mask": tokenized_counterfactual["attention_mask"],
            "labels": tokenized_labels
        }
        
    prompts, edit_instructions, targets, unaffected_attributes, counterfactual_prompts = [], [], [], [], []
    for b in batch:
        prompts.append(b["input_text"])
        edit_instructions.append(b["edit_instruction"])
        targets.append(b["counterfactual_target"])
        unaffected_attributes.append(b["unaffected_attributes"])
        counterfactual_prompts.append(b["counterfactual_input_text"])
        
        
    editor_input_ids = tokenizer(edit_instructions, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    
    returned_dict = {
        "editor_input_ids": editor_input_ids,
        **tokenize_text_inputs(prompts, counterfactual_prompts, targets),
    }
    
    base_prompts_unaffected, counterfactual_prompts_unaffected, targets_unaffected, edit_instructions_unaffected, instance_indices = [], [], [], [], []
    
    for i, attribute_list in enumerate(unaffected_attributes):
        
        for d in attribute_list:
            base_prompts_unaffected.append(d["input_text"])
            targets_unaffected.append(d["target"])
            counterfactual_prompts_unaffected.append(d["counterfactual_input_text"])
                    
        for _ in range(len(attribute_list)):
            edit_instructions_unaffected.append(editor_input_ids[i])
            instance_indices.append(i)
        
    edit_instructions_unaffected = torch.stack(edit_instructions_unaffected)
    instance_indices = torch.tensor(instance_indices)
    
    assert len(base_prompts_unaffected) == len(targets_unaffected)
    
    tokenized_unaffected = tokenize_text_inputs(base_prompts_unaffected, counterfactual_prompts_unaffected, targets_unaffected)
    
    returned_dict["editor_input_ids_unaffected"] = edit_instructions_unaffected
    returned_dict["base_input_ids_unaffected"] = tokenized_unaffected["base_input_ids"]
    returned_dict["base_attention_mask_unaffected"] = tokenized_unaffected["base_attention_mask"]
    returned_dict["source_input_ids_unaffected"] = tokenized_unaffected["source_input_ids"]
    returned_dict["source_attention_mask_unaffected"] = tokenized_unaffected["source_attention_mask"]
    returned_dict["labels_unaffected"] = tokenized_unaffected["labels"]
    returned_dict["instance_indices"] = instance_indices
    
    return returned_dict

batch_size = 16  # 50 or so
data_loader = DataLoader(
    city_train_set, batch_size=batch_size, collate_fn=ravel_collate_fn, shuffle=True
)  # batch_size, collate_fn=collate_fn)
test_data_loader = DataLoader(
    city_test_set, batch_size=batch_size, collate_fn=ravel_collate_fn, shuffle=True
)

for batch in data_loader:
    break


# @torch.compile #Apparently this fails when used inside jupyter notebooks but is fine if i make dedicated scripts
from models.llama3.model import LlamaInterpretor, LlamaInterpretorConfig
from models.utils import EditorModelOutput

class RavelInterpretorHypernetwork(nn.Module):
    # Separating the editor config file, from its base model's configurations
    def __init__(
        self,
        model_name_or_path="/work/frink/models/llama3-8B-HF",
        num_editing_heads=32,
        chop_editor_at_layer=8,
        intervention_layer=10,
        torch_dtype=torch.bfloat16
    ):
        super().__init__()

        self.editor_config = LlamaInterpretorConfig.from_pretrained(model_name_or_path)
        self.editor_config.name_or_path = model_name_or_path
        self.editor_config.torch_dtype = torch_dtype
        self.editor_config.num_editing_heads = num_editing_heads
        self.editor_config.chop_editor_at_layer = chop_editor_at_layer
        self.editor_config.default_intervention_layer = intervention_layer
        self.editor_config._attn_implementation = 'eager'
                
        self.editor_inner = LlamaInterpretor(self.editor_config)
        self.tokenizer = tokenizer

        self.residual_cache = None
        self.opt = None
        self.training_loss = None
        
    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        _pred: EditorModelOutput = self.editor_inner(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids != self.editor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            # output_target_hidden_states=True,
        )
        
        if labels is None:
            return {
                "logits": _pred.logits,
                "target_hidden_states": _pred.target_hidden_states,
            }
        else:
            log_prob_predictions = torch.nn.functional.log_softmax(
                _pred.logits.reshape(-1, _pred.logits.shape[-1]),
                dim=1,
            )
            
            labels = labels.reshape(-1)
            assert labels.shape == log_prob_predictions.shape[:-1]
            
            # Only consider the tokens that are not -100 in target_labels
            label_indices = labels != -100
            
            log_prob_predictions = log_prob_predictions[label_indices, :]
            labels = labels[label_indices]
            
            # Compute the cross-entropy loss with masking
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(log_prob_predictions, labels.long())
            
            return {
                "logits": _pred.logits,
                "target_hidden_states": _pred.target_hidden_states,
                "loss": loss,
            }
        
    
    # Generate text using the target model, with a new edit application at every step.
    # This is a very slow way to generate text.
    # If you only want to edit first k tokens, use the forward pass instead with stop_editing_index = k
    def inspect_batch_prediction_ouptuts(self, batch):
        self.editor_inner.eval()
        
        with torch.no_grad():
            
            predictions = self.forward(
                editor_input_ids=batch["editor_input_ids"].to("cuda"),
                base_input_ids=batch["base_input_ids"].to("cuda"),
                base_attention_mask=batch["base_attention_mask"].to("cuda"),
                source_input_ids=batch["source_input_ids"].to("cuda"),
                source_attention_mask=batch["source_attention_mask"].to("cuda"),
            )
            
            batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
            batch_full_output = self.tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
            
            batch_output = []
            correct = 0
            
            for label, pred_ids in zip(batch["labels"].to("cuda"), batch_pred_ids):
                
                output_idx = label != -100
                label = label[output_idx]
                pred_ids = pred_ids[output_idx]
                batch_output.append(
                    self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                )             
                
                correct += torch.sum(label == pred_ids) == torch.numel(label)
            
        return {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "n_correct": correct,
        }
    
    def eval_accuracy(self, test_loader, use_unaffected=False):
        
        self.editor_inner.eval()
        test_loss = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"].to("cuda"),
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    labels=batch["labels"].to("cuda"),
                )
                test_loss.append(predictions["loss"].item())
                
                batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
                
                if use_unaffected:
                    unaffected_prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids_unaffected"].to("cuda"),
                        base_input_ids=batch["base_input_ids_unaffected"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask_unaffected"].to("cuda"),
                        source_input_ids=batch["source_input_ids_unaffected".to("cuda")],
                        source_attention_mask=batch["source_attention_mask_unaffected"].to("cuda"),
                        labels=batch["labels_unaffected"].to("cuda"),
                    )
                    batch_pred_ids_unaffected = torch.argmax(unaffected_prediction["logits"], dim=-1)
                    
                    correct_unaffected = []
                    
                    for label, pred_ids in zip(batch["labels_unaffected"].to("cuda"), batch_pred_ids_unaffected):
                        output_idx = label != -100
                        label = label[output_idx]
                        pred_ids = pred_ids[output_idx]
                        correct_unaffected.append((torch.sum(label == pred_ids) == torch.numel(label)))
                    
                    correct_unaffected = torch.stack(correct_unaffected)
                    
                    test_loss[-1] += unaffected_prediction["loss"].item()
                
                for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                    output_idx = label != -100
                    label = label[output_idx]
                    pred_ids = pred_ids[output_idx]
                    
                    is_correct = (torch.sum(label == pred_ids) == torch.numel(label)).item()
                    if use_unaffected:
                        # indices of the position which value is i
                        instance_idx = torch.nonzero(batch["instance_indices"].to("cuda") == i).squeeze()
                        # check if all the unaffected positions were correct
                        # if not, set is_correct to False
                        is_correct = is_correct and all(correct_unaffected[instance_idx])
                        
                    correct += is_correct
                    total += 1
        
        print(f"Accuracy: {correct} / {total}")
        return correct / total, sum(test_loss) / len(test_loss)
             

    def run_train(
        self,
        train_loader,
        test_loader=None,
        epochs=1,
        eval_per_steps: int = None,
        use_unaffected = False
    ):
        trainable_parameters = []
        for name, param in self.named_parameters():
            if "target_model" not in name:
                trainable_parameters.append(param)
                
        self.opt = optim.AdamW(trainable_parameters, lr=lr, weight_decay=0.01)  # usually: lr = 5e-5. 1e-3 worked well!

        for epoch in range(epochs):
            # Create a tqdm progress bar
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                disable=True,
            ) as pbar:
                num_datapoints_in_epoch = 0
                epoch_train_loss = 0
                epoch_gradient_norm = 0
                # Train loop
                batch_index = -1  # index of first batch will be 0

                for step, batch in enumerate(
                    train_loader
                ):  
                    if step % eval_per_steps == 0:
                        # Evaluate the model
                        accuracy, test_loss = self.eval_accuracy(
                            test_loader, use_unaffected=use_unaffected
                        )
                        
                        if wandb.run:
                            wandb.log(
                                {
                                    "test_average": test_loss,
                                    "test_accuracy": accuracy,
                                }
                            )
                        
                    batch_index += 1
                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size
                    self.opt.zero_grad()

                    self.prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"].to("cuda"),
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        source_input_ids=batch["source_input_ids"].to("cuda"),
                        source_attention_mask=batch["source_attention_mask"].to("cuda"),
                        labels=batch["labels"].to("cuda"),
                    )

                    self.prediction_loss = self.prediction["loss"]
                    
                    if use_unaffected:
                        self.prediction_loss += self.forward(
                            editor_input_ids=batch["editor_input_ids_unaffected"].to("cuda"),
                            input_ids=batch["input_ids_unaffected"].to("cuda"),
                            attention_mask=batch["attention_mask_unaffected"].to("cuda"),
                            labels=batch["labels_unaffected"].to("cuda"),
                        )["loss"]

                    # Compute the total loss and backpropagate
                    self.training_loss = self.prediction_loss
                    self.training_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), max_grad_clip
                    )  # just implemented this! dunno if a cap of 1 to large, so I'm messing with reducing it

                    # Check for nan gradients
                    # if check_nan_gradients(self):
                    #     break

                    # Backwards step
                    self.opt.step()

                    # metrics
                    epoch_train_loss += self.training_loss.item() * current_batch_size
                    gradients = [
                        p.grad.view(-1) for p in self.parameters() if p.grad is not None
                    ]
                    all_gradients = torch.cat(gradients)
                    gradient_norm = torch.norm(all_gradients).item()
                    epoch_gradient_norm += gradient_norm * current_batch_size

                    metrics = {
                        "step": step * (epoch + 1),
                        "train_batch_total_loss": self.training_loss.item(),
                        "train_batch_prediction_loss": self.prediction_loss.item(),
                        "train_batch_gradient_norm": gradient_norm,
                    }

                    if wandb.run:
                        wandb.log(metrics)
                    if step % 5 == 0:
                        print(metrics)

                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!

                    # Check if it's time to save a checkpoint
                    current_time = time.time()
                    # first loop initialization
                    if batch_index == 0 and epoch == 0:
                        last_checkpoint_time = -100000

                if wandb.run:
                    wandb.log(
                        {
                            "epoch_train_total_loss": epoch_train_loss
                            / num_datapoints_in_epoch,
                            "gradient_norm": epoch_gradient_norm
                            / num_datapoints_in_epoch,
                        }
                    )
            # Save the final model
            torch.save(self.state_dict(), "final_model.pt")




hypernetwork = RavelInterpretorHypernetwork()
hypernetwork = hypernetwork.to("cuda")

# current problem: 1728 / 30864
hypernetwork.run_train(
    train_loader=data_loader,
    test_loader=test_data_loader,
    epochs=15,
    eval_per_steps = 50,
    use_unaffected=False
)
wandb.finish()