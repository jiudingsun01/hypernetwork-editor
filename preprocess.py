from transformers import AutoTokenizer, LlamaForCausalLM
from src.data_utils import generate_ravel_dataset, get_ravel_collate_fn, filter_dataset
from datasets import Dataset, DatasetDict
import torch


SANITY_CHECK_TEMPLATES = {
    "city": {
        "Language": ["People in %s usually speak"],
        "Country": ["%s is in the country of"],
        "Continent": ["%s is in the continent of"]
    }
}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-HF", torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = LlamaForCausalLM.from_pretrained("/work/frink/models/llama3-8B-HF").to("cuda")
    
    
    city_train_set = generate_ravel_dataset(tokenizer, n_samples=100000, root_path="./data/ravel/ravel_raw", split="train", disentangling=True)
    city_test_set = generate_ravel_dataset(tokenizer, n_samples=5000, root_path="./data/ravel/ravel_raw", split="test", disentangling=True)
    
    filtered_city_train_set = filter_dataset(model, tokenizer, city_train_set, disentangling=True, batch_size=32)
    filtered_city_test_set = filter_dataset(model, tokenizer, city_test_set, disentangling=True, batch_size=32)
    
    city_dataset = DatasetDict({
        "train": filtered_city_train_set,
        "test": filtered_city_test_set
    })
    city_dataset.save_to_disk("./data/ravel/city_disentangling")
    