import os
import json


huntington_config = [
    "--partition=177huntington",
    "--nodes=1",
    "--gres=gpu:1",
    "--ntasks-per-node=1",
    "--mem=32GB",
    "--time=24:00:00"
]

frink_config = [
    "--partition=frink",
    "--nodes=1",
    "--gres=gpu:1",
    "--ntasks-per-node=1",
    "--mem=32GB",
    "--time=24:00:00"
]


discovery_v100_config = [
    "--partition=gpu",
    "--nodes=1",
    "--gres=gpu:v100-sxm2:1",
    "--ntasks-per-node=1",
    "--mem=16GB",
    "--time=8:00:00"
]


discovery_a100_config = [
    "--partition=multigpu",
    "--nodes=1",
    "--gres=gpu:a100:1",
    "--ntasks-per-node=1",
    "--mem=16GB",
    "--time=8:00:00"
]


def create_job_script(script_path: str, python_name: str, job_name: str, job_output: str, server_configs: list = huntington_config, dependency=True, **script_args):
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output={job_output}\n")
        for config in server_configs:
            f.write(f"#SBATCH {config}\n")
        if dependency:
            f.write(f"#SBATCH --dependency=singleton\n")
        f.write("\n\n")
        f.write("source /work/frink/sun.jiu/miniconda3/bin/activate\n")
        f.write("cd /work/frink/sun.jiu/hypernetwork-editor\n")
        f.write("conda activate subspace\n")
        f.write("\n\n")
        
        f.write(f"python {python_name} ")
        for arg, value in script_args.items():
            f.write(f"--{arg} {value} ")
        f.close()