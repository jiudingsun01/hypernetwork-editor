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
from src.data_utils import get_ravel_collate_fn, generate_ravel_dataset


from transformers import AutoTokenizer


def run_experiment(intervention_layer=6):
    
    wandb.init(
        project="hypernetworks-interp",
        name=f"test_run",
        config={"targetmodel": "llama3-8b", "editormodel": "llama3-8b", "dataset": "ravel"},
    )  
    
    tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-HF")
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    city_dataset = load_from_disk("./data/ravel/city_one_token")
    train_set = city_dataset["train"]
    test_set = city_dataset["test"]

    collate_fn = get_ravel_collate_fn(tokenizer)

    batch_size = 32  # 50 or so
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )  # batch_size, collate_fn=collate_fn)
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    # @torch.compile #Apparently this fails when used inside jupyter notebooks but is fine if i make dedicated scripts
    from src.llama3.modules import LlamaInterpretor, LlamaInterpretorConfig
    from src.utils import EditorModelOutput
    from src.llama3.model import RavelInterpretorHypernetwork


    hypernetwork = RavelInterpretorHypernetwork(
        model_name_or_path="/work/frink/models/llama3-8B-HF",
        num_editing_heads=16,
        intervention_layer=intervention_layer
    )

    hypernetwork = hypernetwork.to("cuda")

    # current problem: 1728 / 30864
    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=50,
        eval_per_steps = 100,
        disentangling=False,
        save_dir=f"./ravel_city_layer_{intervention_layer}_one_token",
        weight_decay=0.01, 
        lr=3e-5
    )

    wandb.finish()
    
if __name__ == "__main__":
    run_experiment()