from sbatch_utils import *
import os


if __name__ == "__main__":
    
    scripts_paths = []
    
    for i, attribute in enumerate(["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"]):
        
        
        job_idx = (i * 2) % 4 
        script_path = f"./scripts/train_{attribute}.sh"
        python_name = "train.py"
        job_name = f"training_{job_idx}"
        job_output = f"./logs/train_{attribute}.out"
        server_configs=huntington_config
        
        save_dir = "city_" + attribute
        wandb_project = "hypernetwork-interp-city-" + attribute
        dataset_path = "./data/ravel/city_" + attribute
        disentangling = ""
        
        create_job_script(
            script_path, python_name, job_name, job_output, server_configs, dependency=True,
            wandb_project=wandb_project, save_dir=save_dir, dataset_path=dataset_path, disentangling=disentangling
        )
        
        scripts_paths.append(script_path)
        
        
        job_idx = (i * 2 + 1) % 4 
        script_path = f"./scripts/train_{attribute}_das.sh"
        python_name = "train.py"
        job_name = f"training_{job_idx}"
        job_output = f"./logs/train_{attribute}_das.out"
        server_configs=huntington_config
        
        save_dir = "city_" + attribute + "_das"
        wandb_project = "hypernetwork-interp-city-" + attribute
        dataset_path = "./data/ravel/city_" + attribute
        disentangling = ""
        use_das_intervention = ""
        das_dimension = 256
        
        create_job_script(
            script_path, python_name, job_name, job_output, server_configs, dependency=True,
            wandb_project=wandb_project, save_dir=save_dir, dataset_path=dataset_path, disentangling=disentangling, use_das_intervention=use_das_intervention, das_dimension=das_dimension
        )
        
        scripts_paths.append(script_path)
        
    for script_path in scripts_paths:
        os.system(f"sbatch {script_path}")
        
        