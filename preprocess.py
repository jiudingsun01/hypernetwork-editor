from transformers import AutoTokenizer, LlamaForCausalLM
from src.data_utils import *
from datasets import Dataset, DatasetDict
import torch
import argparse
import os


SANITY_CHECK_TEMPLATES = {
    "city": {
        "Language": ["People in %s usually speak"],
        "Country": ["%s is in the country of"],
        "Continent": ["%s is in the continent of"]
    }
}

def preprocess(
    model_name_or_path="/work/frink/models/llama3-8B-HF",
    n_train_samples=1000,
    n_test_samples=1000,
    disentangling=True,
    domains_excluded_attributes=[[]],
    target_attributes=[["Country"]],
    save_dir=None,
    filtering=True
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to("cuda")
    
    
    city_train_set = generate_ravel_prefix_suffix_dataset(
        tokenizer, n_samples=n_train_samples, split="train", disentangling=disentangling,
        domains_excluded_attributes=domains_excluded_attributes, target_attributes=target_attributes
    )
    city_test_set = generate_ravel_prefix_suffix_dataset(
        tokenizer, n_samples=n_test_samples, split="test", disentangling=disentangling,
        domains_excluded_attributes=domains_excluded_attributes, target_attributes=target_attributes
    )
    
    if filtering:
        city_train_set = filter_dataset(model, tokenizer, city_train_set, disentangling=disentangling, batch_size=32, prefix_and_suffix=True)
        city_test_set = filter_dataset(model, tokenizer, city_test_set, disentangling=disentangling, batch_size=32, prefix_and_suffix=True)
    
    city_dataset = DatasetDict({
        "train": city_train_set,
        "test": city_test_set
    })
    
    if save_dir is not None:
        city_dataset.save_to_disk(os.path.join("./data/ravel/", save_dir))
        
    return city_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="/work/frink/models/llama3-8B-HF")
    parser.add_argument("--n_train_samples", type=int, default=200000)
    parser.add_argument("--n_test_samples", type=int, default=10000)
    parser.add_argument("--not_disentangling", action="store_false")
    parser.add_argument("--domains_excluded_attributes", type=json.loads, default=[])
    parser.add_argument("--target_attributes", type=str, default="Country")
    parser.add_argument("--save_dir", type=str, default="city_country")
    parser.add_argument("--not_filtering", action="store_false")
    
    args = parser.parse_args()
    
    preprocess(
        model_name_or_path=args.model_name_or_path,
        n_train_samples=args.n_train_samples,
        n_test_samples=args.n_test_samples,
        disentangling=args.not_disentangling,
        domains_excluded_attributes=[args.domains_excluded_attributes],
        target_attributes=[[args.target_attributes]],
        save_dir=args.save_dir,
        filtering=args.not_filtering
    )