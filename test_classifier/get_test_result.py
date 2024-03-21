import torch
import yaml
import argparse
import matplotlib.pyplot as plt

# Custom packages
from test_classifier.membrane_extractor import extract_corrected_membrane
from test_classifier.classify_membrane import classify_membrane

def get_test_result(image_path, model_seg, model_cla, show_bool=False):

    # Load required arguments
    args_seg_path = "segmentation/config_seg.yaml"
    args_cla_path = "classification/config_cla.yaml"
    with open(args_seg_path) as f:
        args_seg = argparse.Namespace(**yaml.safe_load(f))
    with open(args_cla_path) as f:
        args_cla = argparse.Namespace(**yaml.safe_load(f))
    # Extract membrane
    membrane, *_ = extract_corrected_membrane(args_seg, image_path, model_seg)
    # Predict result
    classify_membrane(args_cla, membrane, model_cla, show_bool=show_bool)

if __name__ == '__main__':
    
    import os
    from segmentation import model_seg
    from classification import model_cla

    # Image paths
    kit_id = 'aconag'
    data_path = os.path.join('segmentation/data_seg/', kit_id, f'{kit_id}_images')
    image_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
    # Load pre-trained segmentation model
    state_path = 'segmentation/models_seg/2024-03-06_13.04.44_best.state'
    state = torch.load(state_path, map_location='cpu')
    model_s = model_seg.get_segmentation_model(3, 256)
    model_s.load_state_dict(state['model_state'])
    # Load pre-trained classification model
    state_path = 'classification/models_cla/2024-03-03_18.28.26_best.state'
    model_c = model_cla.ClassificationModel.from_pretrained(state_path)
    # Predict result
    image_path = "data/internet_images/internet_3.webp" #image_paths[35]
    get_test_result(image_path, model_s, model_c, show_bool=True)
