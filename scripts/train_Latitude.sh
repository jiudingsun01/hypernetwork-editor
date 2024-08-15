#!/bin/bash
#SBATCH --job-name=training_2
#SBATCH --output=./logs/train_Latitude.out
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


python train.py --wandb_project hypernetwork-interp-city-Latitude --save_dir city_Latitude --dataset_path ./data/ravel/city_Latitude --disentangling  