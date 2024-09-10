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
from src.data_utils import get_ravel_collate_fn, generate_ravel_dataset_from_filtered
import argparse


from transformers import AutoTokenizer


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    inference_modes=["default", "bidding_argmax"],
    intervention_layer=15,
    subspace_module="ReflectSelect",
    model_name_or_path="/home/ubuntu/llama3-8b",
    batch_size=8,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    save_dir=None,
    das_dimension=None,
    n_epochs=1,
    n_samples=20000,
    train_test_split=0.95,
    lr=3e-5,
    weight_decay=0.01,
    eval_per_steps=100,
    checkpoint_per_steps=500,
    domain="city",
    filtered_dataset_path=None,
    isolate_attributes=["Country"],
    target_attributes=["Continent"],
    test_path=None,
    train_path=None,
):
    
    if filtered_dataset_path is None:
        assert train_path is not None and test_path is not None
    if save_dir is not None:
        save_dir = os.path.join("./models", save_dir)
        
    
        
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "targetmodel": model_name_or_path, 
                "editormodel": model_name_or_path, 
                "dataset": "ravel",
                "intervention_layer": intervention_layer,
                "subspace_module": subspace_module,
                "source_suffix_visibility": source_suffix_visibility,
                "base_suffix_visibility": base_suffix_visibility,
                "das_dimension": das_dimension,
                "domain": domain,
                "isolate_attributes": isolate_attributes,
                "target_attributes": target_attributes,
            },
        )
        
    if "default" in inference_modes:
        inference_modes.remove("default")
        inference_modes.append(None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if filtered_dataset_path is not None:
        city_dataset = generate_ravel_dataset_from_filtered(
            n_samples=n_samples,
            domain=domain,
            filtered_dataset_path=filtered_dataset_path,
            isolate_attributes=isolate_attributes,
            target_attributes=target_attributes
        )
        city_dataset = city_dataset.shuffle()
        train_test_split = int(n_samples * train_test_split)
        train_set = city_dataset.select(range(train_test_split))
        test_set = city_dataset.select(range(train_test_split, n_samples))
    else:
        train_set = load_from_disk(train_path)
        test_set = load_from_disk(test_path)
                
    collate_fn = get_ravel_collate_fn(
        tokenizer, 
        source_suffix_visibility=source_suffix_visibility, 
        base_suffix_visibility=base_suffix_visibility, 
        add_space_before_target=True
    )
    
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.llama3.model import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(
        model_name_or_path=model_name_or_path,
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        subspace_module=subspace_module,
        das_dimension=das_dimension,
    )

    hypernetwork = hypernetwork.to("cuda")

    # current problem: 1728 / 30864
    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        inference_modes=inference_modes,
        epochs=n_epochs,
        checkpoint_per_steps = checkpoint_per_steps,
        eval_per_steps = eval_per_steps,
        save_dir=save_dir,
        weight_decay=weight_decay, 
        lr=lr
    )

    if log_wandb:
        wandb.finish()
        
    if save_dir is not None:
        train_set.save_to_disk(os.path.join(save_dir, "train"))
        test_set.save_to_disk(os.path.join(save_dir, "test"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="hypernetworks-interpretor")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--model_name_or_path", type=str, default="/home/ubuntu/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--source_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--test_path", type=str, default= "./data/ravel/baseline_debug_test")
    parser.add_argument("--train_path", type=str, default= "./data/ravel/baseline_debug_train")
    
    parser.add_argument("--filtered_dataset_path", type=str, default=None)
    
    # if filtered_dataset_path is not None:
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--train_test_split", type=int, default=0.97)
    parser.add_argument("--domain", type=str, default="city")
    parser.add_argument('--isolate_attributes', nargs='+', default=["Country"])
    parser.add_argument('--target_attributes', nargs='+', default=["Country", "Continent"])
    
    parser.add_argument('--inference_modes', nargs='+', default=["default", "bidding_argmax"])
    
    
    # if None, use Boundless DAS
    parser.add_argument('--subspace_module', default="DAS", choices=[None, "DAS", "BoundlessDAS", "MaskSelect", "ReflectSelect"])
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)