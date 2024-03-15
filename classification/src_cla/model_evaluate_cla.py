"""
File containing the EvalClassification class, the main class for evaluating a classification model.
"""

import os
import argparse
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
        
        # Split configuration file
        self.data_args = argparse.Namespace(**args.data_args)
        self.inference_args = argparse.Namespace(**args.inference_args)

        # Align input device with model's device
        if self.device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.model.to(device)
        # Set model in evaluation mode
        self.model.eval()
    
    def evaluate(self, kit_id, data_mode='test', save_bool=False, save_filename=None):
        """
        Main function. Run inference on all images in the corresponding set, plot confussion matrix and return 
        misclassified zones and a dataframe with information on the misclassified examples.
        """

        if data_mode not in ['train', 'test', 'val']:
            raise ValueError("data_mode must be 'train', 'val', or 'test'!")

        # Initialize data loader
        loader = init_dataloader(
            self.data_args, kit_id, data_mode, 
            self.n_batches, 
            self.n_workers, 
            shuffle=False, 
            shots=None, 
            transform=self.transform)

        # Run inference on all data, returning predictions and metrics
        inference_df = self.run_inference(loader, data_mode)
        
        # Define dataframe of misclassified examples and get the corresponding zones
        misclassified_df = inference_df[inference_df['targets'] != inference_df['predictions']]
        misclassified_ndx = misclassified_df.index.tolist()
        try:
            misclassified_zones = torch.stack([loader.dataset[i//2][0][j] for i, j in zip(misclassified_ndx, misclassified_df.zone_ndx)])
        except:  # Catch empty list (no misclassified)
            misclassified_zones = []
        
        # After extracting zones, set filenames column as the index
        misclassified_df.set_index('filenames', drop=True, inplace=True)

        # Save metrics to csv file
        if save_bool:
            if save_filename is not None:
                save_path = os.path.join(self.inference_args.output_dir, f'{kit_id}_{data_mode}_{save_filename}_misclassified.csv')
            else:
                save_path = os.path.join(self.inference_args.output_dir, f'{kit_id}_{data_mode}_misclassified.csv')

            # Check if filename file already exists.
            if os.path.exists(save_path):
                overwrite = input(f"File {save_path} already exists! Do you want to overwrite it? (y/n): ")
                while overwrite not in ['y', 'n']:
                    overwrite = input(f"Introduce 'y' or 'n': ")        
                if overwrite == 'n':
                    print('The file was not overwritten :)')
                if overwrite == 'y':
                    misclassified_df.to_csv(save_path)
            else:
                misclassified_df.to_csv(save_path)

        # Compute confussion matrix and display it
        confusion_df = pd.crosstab(inference_df.targets, inference_df.predictions)
        error_rate = len(misclassified_df)/len(inference_df)

        plt.figure(figsize=(8,4))
        sns.heatmap(confusion_df, annot=True, fmt='d')
        plt.title(f'Error = {error_rate*100:.3f}% ({len(misclassified_df)}/{len(inference_df)})')
        plt.show()
            
        return misclassified_zones, misclassified_df
    

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


        