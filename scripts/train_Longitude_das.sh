#!/bin/bash
#SBATCH --job-name=training_2
#SBATCH --output=./logs/train_Longitude_das.out
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


python train.py --save_dir city_Longitude_L15 --wandb_project hypernetwork-autointerp --isolate_attributes Country Continent Language Latitude --target_attributes Longitude --das_dimension 128 