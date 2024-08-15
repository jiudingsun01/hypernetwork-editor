from sbatch_utils import *
import os


if __name__ == "__main__":
    
    scripts_paths = []
    
    for i, attribute in enumerate(["Country", "Continent", "Language", "Timezone", "Longitude", "Latitude"]):
        job_idx = i % 3
        script_path = f"./scripts/preprocess_{attribute}.sh"
        python_name = "preprocess.py"
        job_name = f"preprocess_{job_idx}"
        job_output = f"./logs/preprocess_{attribute}.out"
        server_configs=huntington_config
        
        target_attributes = attribute
        save_dir = "city_" + attribute
        
        
        create_job_script(
            script_path, python_name, job_name, job_output, server_configs, dependency=True,
            target_attributes=target_attributes, save_dir=save_dir
        )
        
        scripts_paths.append(script_path)
        
    for script_path in scripts_paths:
        os.system(f"sbatch {script_path}")
        
        