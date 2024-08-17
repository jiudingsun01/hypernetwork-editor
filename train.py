import torch
from torch import compile
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os
import time
import sys
import wandb
import random
import numpy as np
import json
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from src.data_utils import get_ravel_prefix_suffix_collate_fn, generate_ravel_prefix_suffix_dataset, generate_ravel_dataset_from_filtered
import argparse


from transformers import AutoTokenizer


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    intervention_layer=15,
    no_das=False,
    model_name_or_path="/work/frink/models/llama3-8B-HF",
    dataset_path="./data/ravel/city_country_prefix_and_suffix_disentangling",
    batch_size=8,
    no_disentangling=False,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    save_dir=None,
    das_dimension=None
):
    if save_dir is not None:
        save_dir = os.path.join("./models", save_dir)
        
    use_das_intervention = not no_das
    disentangling = not no_disentangling
        
    if log_wandb:
        run_name = f"L{intervention_layer}"
        if disentangling:
            run_name += "-Disentangling"
        if use_das_intervention:
            run_name += "-DAS"
            
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={"targetmodel": "llama3-8b", "editormodel": "llama3-8b", "dataset": "ravel-city"},
        )  
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    """
    city_dataset = load_from_disk(dataset_path)
    train_set = city_dataset["train"]
    test_set = city_dataset["test"]
    collate_fn = get_ravel_prefix_suffix_collate_fn(tokenizer, disentangling=disentangling, source_suffix_visibility=source_suffix_visibility, base_suffix_visibility=base_suffix_visibility)
    """
    
    city_dataset = generate_ravel_dataset_from_filtered(
        tokenizer=tokenizer,
        n_samples=20000,
    )
    city_dataset = city_dataset.shuffle()
    train_set = city_dataset.select(range(19000))
    test_set = city_dataset.select(range(19000, 20000))
    collate_fn = get_ravel_prefix_suffix_collate_fn(tokenizer, disentangling=False, source_suffix_visibility=True, base_suffix_visibility=base_suffix_visibility, add_space_before_target=False)
    
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )  # batch_size, collate_fn=collate_fn)
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.llama3.modules import LlamaInterpretor, LlamaInterpretorConfig
    from src.utils import EditorModelOutput
    from src.llama3.model import RavelInterpretorHypernetwork


    hypernetwork = RavelInterpretorHypernetwork(
        model_name_or_path=model_name_or_path,
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        das_intervention=use_das_intervention,
        das_dimension=das_dimension,
    )

    hypernetwork = hypernetwork.to("cuda")

    # current problem: 1728 / 30864
    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=10,
        checkpoint_per_steps = 500,
        eval_per_steps = 75,
        disentangling=False,
        save_dir=save_dir,
        weight_decay=0.00, 
        lr=3e-5
    )

    if log_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="hypernetworks-interpretor")
    parser.add_argument("--intervention_layer", type=int, default=15)
    parser.add_argument("--no_das", default=False, action="store_false")
    parser.add_argument("--model_name_or_path", type=str, default="/work/frink/models/llama3-8B-HF")
    parser.add_argument("--dataset_path", type=str, default="./data/ravel/city_Country")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_disentangling", default=False, action="store_false")
    parser.add_argument("--source_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    
    # if None, use Boundless DAS
    parser.add_argument("--das_dimension", type=int, default=None)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)