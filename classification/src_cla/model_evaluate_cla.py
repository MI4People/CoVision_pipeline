"""
File containing the EvalClassification class, the main class for evaluating a classification model.
"""

import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from dataset_cla import MembraneZonesDataset
from dataloader_cla import init_dataloader
from transformations_cla import PreprocessMembraneZones


class EvalClassification():
    def __init__(self, args, model, device=None, n_batches=16, n_workers=0):
        
        # Assign attributes
        self.args = args
        self.model = model
        self.device = device
        self.n_batches = n_batches
        self.n_workers = n_workers

        # Transformation
        self.transform = PreprocessMembraneZones()

        # Align input device with model's device
        if self.device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.model.to(device)
        # Set model in evaluation mode
        self.model.eval()
    
    def evaluate(self, kit_id, data_mode='test', save_bool=False, save_path=None):
        """
        Main function. Run inference on all images in the corresponding set, plot confussion matrix and return 
        misclassified zones and a dataframe with information on the misclassified examples.
        """

        if data_mode not in ['all', 'train', 'val', 'test']:
            raise ValueError("data_mode must be 'all', 'train', 'val', or 'test'!")
        
        if isinstance(kit_id, str):
            kit_id = [kit_id]

        # Initialize data loader
        loader = init_dataloader(
            self.args, kit_id, data_mode, 
            self.n_batches, 
            self.n_workers, 
            shuffle=False, 
            shots=[None]*len(kit_id), 
            transform=self.transform)

        # Run inference on all data, returning predictions and metrics
        inference_df = self.run_inference(loader, data_mode)
        n_all = len(inference_df)
        
        # Define dataframe of misclassified examples and get the corresponding zones
        misclassified_df = inference_df[inference_df['targets'] != inference_df['predictions']]
        n_wrong = len(misclassified_df)
        misclassified_ndx = misclassified_df.index.tolist()
        try:
            misclassified_zones = torch.stack([loader.dataset[i//2][0][j] for i, j in zip(misclassified_ndx, misclassified_df.zone_ndx)])
        except:  # Catch empty list (no misclassified)
            misclassified_zones = []
        
        # After extracting zones, set filenames column as the index
        misclassified_df.set_index('filenames', drop=True, inplace=True)

        # Compute confussion matrix and accuracy
        confusion_df = pd.crosstab(inference_df.targets, inference_df.predictions)
        error_rate = n_wrong/n_all
        accuracy = 1 - error_rate

        # Create confussion matrix figure
        plt.figure(figsize=(8,4))
        sns.heatmap(confusion_df, annot=True, fmt='d')
        plt.title(f'Error = {error_rate*100:.3f}% ({n_wrong}/{n_all}) for {"-".join(kit_id)} kit(s), {data_mode} mode')

        # Save misclassified_df, matrix image and accuracy
        if save_bool:
            self.save_results(save_path, misclassified_df, accuracy, n_wrong, n_all, kit_id, data_mode)
        
        # Display figure (after saving it!)
        plt.show()

        return misclassified_zones, misclassified_df, accuracy
    

    def run_inference(self, loader, data_mode):
        """
        Run inference on entire dataloader and return dataframe with targets and predictions
        """

        # membrane filenames and zone index [0, 1, 0, 1, ...] for later tracking of misclassified
        zone_ndx = [i for j in range(len(loader.dataset)) for i in range(2)]
        filenames = [f'{name}' for name in loader.dataset.filenames for i in range(2)]
        # Keep track of targets and predictions for each zone
        outputs = {'filenames': filenames, 'zone_ndx':zone_ndx, 'targets': [], 'predictions': []}
        zones = []
        
        # To log fancy progress bar
        loop = tqdm(loader)
        loop.set_description(f"Running inference")
        
        for x, y in loop:
            # Send to device
            x = x.to(torch.device('cpu'))
            y = y.to(torch.device('cpu')).squeeze().to(int).tolist()
            # Forward pass
            y_pred = self.model(x)
            # Binarize prediction
            binary_pred = (y_pred > 0.5).squeeze().to(int).tolist()
            # Store targets and predictions
            outputs['targets'].extend(y)
            outputs['predictions'].extend(binary_pred)
            loop.update()
        
        outputs_df = pd.DataFrame.from_dict(outputs)

        return outputs_df

    @staticmethod
    def save_results(save_path, misclassified_df, accuracy, n_wrong, n_all, kit_id, data_mode):
        """
        Save the misclassified dataframe as a csv file, the confusion matrix as an image, 
        and the accuracy and number of examples (n_wrong and n_all) in a yaml file.
        """

        # Create folder if it doesn't exist
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Convert kit_id list into concatenated strings
        kit_id = '-'.join(kit_id)
        # Create the three paths and ask whether to overwrite them or not
        df_path = os.path.join(save_path, f'{kit_id}_{data_mode}_misclassified.csv')
        img_path = os.path.join(save_path, f'{kit_id}_{data_mode}_confussion_matrix.png')
        info_path = os.path.join(save_path, f'{kit_id}_{data_mode}_info.yaml')

        for _path in [df_path, img_path, info_path]:
            if os.path.exists(_path):
                overwrite = input(f"File {_path} already exists! Do you want to overwrite it? (y/n): ")
                while overwrite not in ['y', 'n']:
                    overwrite = input(f"Introduce 'y' or 'n': ")
                if overwrite == 'n':
                    print('The file was not overwritten :)')
                    continue  # Skip interation
                elif overwrite == 'y':
                    print('The file will be overwritten!')

            # Save
            if _path == df_path:
                misclassified_df.to_csv(df_path)
            elif _path == img_path:
                plt.savefig(img_path)
            elif _path == info_path:
                with open(info_path, 'w') as f:
                    yaml.dump({'accuracy': accuracy, 'n_wrong': n_wrong, 'n_all': n_all}, f)
                        
        
        
        