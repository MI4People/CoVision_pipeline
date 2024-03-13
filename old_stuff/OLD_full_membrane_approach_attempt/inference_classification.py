import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def run_inference_dataset(dataset, model):
    """
    Compute accuracy of entire dataset
    """
    
    # Send model to cpu and set it to evaluation mode
    model.to(torch.device('cpu'))
    model.eval()
    
    # Build Dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    n_loader = len(loader)
    
    # Store targets and predictions
    outputs = {'targets':[], 'predictions': []}
    
    print('Running Inference...')

    for x, y in loader:
        # Send to device
        x = x.to(torch.device('cpu'))
        y = y.to(torch.device('cpu')).squeeze().to(int).tolist()
        # Forward pass
        y_pred = model(x)
        # Binarize prediction
        binary_pred = (y_pred > 0.5).squeeze().to(int).tolist()
        # Store targets and predictions
        outputs['targets'].extend(y)
        outputs['predictions'].extend(binary_pred)
    
    outputs_df = pd.DataFrame.from_dict(outputs)
    return outputs_df