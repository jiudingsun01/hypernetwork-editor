from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
from src.data_utils import generate_ravel_dataset, get_ravel_collate_fn, filter_dataset

from transformers import AutoTokenizer, LlamaForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-HF")
    model = LlamaForCausalLM.from_pretrained("/work/frink/models/llama3-8B-HF", torch_dtype=torch.float16)
    model = model.cuda()
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = generate_ravel_dataset(
        40000,
        isolate_attributes=["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"],
        target_attributes=["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"],
        template_split="both",
        entity_split="train",
    )
    
    train_dataset = filter_dataset(model, tokenizer, train_dataset, batch_size=8)
    train_dataset = filter_dataset(model, tokenizer, train_dataset, batch_size=8)
    train_dataset = filter_dataset(model, tokenizer, train_dataset, batch_size=8)
    
    train_dataset.save_to_disk("./data/ravel/mixed_train")
    
    test_dataset = generate_ravel_dataset(
        4000,
        isolate_attributes=["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"],
        target_attributes=["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"],
        template_split="both",
        entity_split="test",
    )
    
    test_dataset = filter_dataset(model, tokenizer, test_dataset, batch_size=8)
    test_dataset = filter_dataset(model, tokenizer, test_dataset, batch_size=8)
    test_dataset = filter_dataset(model, tokenizer, test_dataset, batch_size=8)
    
    test_dataset.save_to_disk("./data/ravel/mixed_test")
    
    
    
    
    