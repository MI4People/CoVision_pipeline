import mlflow
import yaml
from pathlib import Path
import os

def create_mlflow_experiment(experiment_name, mlruns_path):
    """
    Custom create_experiment function which wraps around the mlflow.create_experiment function 
    to modify some of its (undesired) behaviour, such as the artifact location.
    """
    
    # Create experiment and catch already_exist error. Returns experiment_id in both cases
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=mlruns_path.as_uri())

        # Change artifact location so artifacts lie inside the experiment folder (as they should!)
        # Read
        with open(Path.joinpath(mlruns_path, experiment_id, 'meta.yaml'), 'r') as f:
            data = yaml.safe_load(f)
        # Modify artifact location
        data['artifact_location'] = os.path.join(data['artifact_location'], experiment_id)
        with open(Path.joinpath(mlruns_path, experiment_id, 'meta.yaml'), 'w') as f:
            yaml.dump(data, f)

    except Exception as e:
        print(e)
        experiment_id = [ex.experiment_id for ex in mlflow.search_experiments() if ex.name == experiment_name][0]
        
    return experiment_id