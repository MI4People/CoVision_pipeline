import os
import yaml
from pathlib import Path
import mlflow

def initiate_mlflow_server(using_colab):
  """
  Initiate mlflow sever to default location http://127.0.0.1:5000. The local 
  mlruns folder is located inside the models folder.
  """

  # Initiate server by running command from bash (the mlrun folder should be created in the models directory)

  if using_colab:

    os.system("mlflow server &")  # By default: host = localhost (127.0.0.1), port = 5000

    from pyngrok import ngrok  # Create remote tunnel using ngrok.com to allow local port access

    # Terminate open tunnels if exist
    ngrok.kill()

    # Setting the authtoken (should be stored in local yml file, otherwise visit https://dashboard.ngrok.com/auth)
    with open('../../utils/ngrok.yml', 'r') as f:
      NGROK_AUTH_TOKEN = yaml.safe_load(f)['authtoken']
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Open an HTTPs tunnel on port 5000 for http://localhost:5000
    ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)

  else:

    os.system("fuser -k 5000/tcp")  # Terminate open servers
    os.system("mlflow server &")  # By default: host = localhost (127.0.0.1), port = 5000

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


def create_mlflow_experiment(experiment_name, tags):
    """
    Custom create_experiment function which wraps around the mlflow.create_experiment function 
    to modify some of its (undesired) behaviour, such as the artifact location.
    """
    
    # Create experiment and catch already_exist error. Returns experiment_id in both cases
    mlruns_path = Path.cwd().joinpath('mlruns')
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=mlruns_path.as_uri(), tags=tags)
    except Exception as e:
        print(e)
        experiment_id = [ex.experiment_id for ex in mlflow.search_experiments() if ex.name == experiment_name][0]
    
    # Change artifact location so artifacts lie inside the experiment folder (as they should!)
    # Read
    experiment_path = Path.joinpath(mlruns_path, experiment_id)
    with open(Path.joinpath(experiment_path, 'meta.yaml'), 'r') as f:
        data = yaml.safe_load(f)
    # Modify artifact location
    data['artifact_location'] = os.path.join(mlruns_path, experiment_id)
    with open(Path.joinpath(experiment_path, 'meta.yaml'), 'w') as f:
        yaml.dump(data, f)

    # Change the yaml file for each run inside the experiment
    directories = next(os.walk(experiment_path))[1]
    for run_id in directories:
        run_path = os.path.join(experiment_path, run_id)
        if 'meta.yaml' in os.listdir(run_path):
          with open(os.path.join(run_path, 'meta.yaml'), 'r') as f:
            data = yaml.safe_load(f)
          # Modify artifact location
          data['artifact_uri'] = os.path.join(run_path, 'artifacts')
          with open(os.path.join(run_path, 'meta.yaml'), 'w') as f:
              yaml.dump(data, f)
        
    return experiment_id
