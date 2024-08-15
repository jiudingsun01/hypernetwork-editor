#!/bin/bash
#SBATCH --job-name=training_3
#SBATCH --output=./logs/train_Timezone_das.out
#SBATCH --partition=177huntington
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --dependency=singleton


source /work/frink/sun.jiu/miniconda3/bin/activate
cd /work/frink/sun.jiu/hypernetwork-editor
conda activate subspace


python train.py --wandb_project hypernetwork-interp-city-Timezone --save_dir city_Timezone_das --dataset_path ./data/ravel/city_Timezone --disentangling  --use_das_intervention  --das_dimension 256 