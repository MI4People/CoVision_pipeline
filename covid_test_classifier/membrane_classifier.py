"""
TODO:
"""

import torch

# Custom packages
from utils_cla.miscellaneous import crop_zones
from transformations_cla import PreprocessMembraneZones

def classify_membrane(args, membrane, model):
    """
    Takes a raw image of a membrane and log whether it corresponds to a positive, negative or invalid LFA test.
    """

    if model.training:
        model.eval()
    
    # Crop relevant zones
    zones = crop_zones(args, membrane)
    # Apply transformation
    transformation = PreprocessMembraneZones()
    zones = transformation(zones)
    # Run inference
    with torch.no_grad():
        y_pred = model(zones)
    # Binarize prediction
    binary_pred = (y_pred > 0.5).squeeze().to(int).tolist()
    # To string
    pred_str = str(binary_pred[0]) + str(binary_pred[1])
    # Map to diagnosis
    result = args.diagnosis_map[pred_str]
    if result == 1:
        result_str = f'Positive ({pred_str})'
    elif result == 0:
        result_str = f'Negative ({pred_str})'
    elif result == 99:
        result_str = f'Invalid ({pred_str})'

    return result_str

    

