"""
File containing the EvalSegmentation class, the main class for evaluating a segmentation model.
"""

import os
from pathlib import Path
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from utils_seg.miscellaneous import compute_iou_mask, compute_iou_box
from utils_seg.visualization import show_images
from dataset_seg import LFASegmentationDataset
from dataloader_seg import init_dataloader

class EvalSegmentation():
    def __init__(self, args, model, device=None, n_batches=2, n_workers=1):
        
        # Assign attributes
        self.args = args
        self.model = model
        self.device = device
        self.n_batches = n_batches
        self.n_workers = n_workers
        
        # Split configuration file
        self.data_args = argparse.Namespace(**args.data_args)
        self.evaluation_args = argparse.Namespace(**args.evaluation_args)
        
        # Classes excluding background
        self.classes = self.data_args.classes[1:]
        self.class_ids = self.data_args.class_ids[1:]
        self.n_classes = len(self.classes)

        # Align input device with model's device
        if self.device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.model.to(device)
        # Set model in evaluation mode
        self.model.eval()
    
    def evaluate(self, kit_id, data_mode='test', show_bool=False, save_bool=False, save_path=None):
        """
        Main function. Run inference on all images in the corresponding set 
        and save all scores and IoU in a csv file.
        """

        if data_mode not in ['all', 'train', 'test', 'val']:
            raise ValueError("data_mode must be 'all', 'train', 'val', or 'test'!")

        if isinstance(kit_id, str):
          kit_id = [kit_id]

        # Initialize data loader
        loader = init_dataloader(
            self.data_args, kit_id, data_mode, 
            n_batches=self.n_batches, 
            n_workers=self.n_workers,
            shuffle=False,
            shots=[None]*len(kit_id),
            transform=None)

        # Run inference on all data, returning predictions and metrics
        predictions, metrics = self.run_inference(loader, data_mode)
        
        # Show all images and their predictions
        if show_bool:
            images = [img.to('cpu') for img, _ in loader.dataset]
            show_images(images, predictions, metrics)

        # Format metrics as Dataframe
        metrics_df = self.get_metrics(metrics, image_names=loader.dataset.filenames)

        # Save stats to file
        if save_bool:
            self.save_results(save_path, metrics_df, kit_id, data_mode)

        # Log mean value of the metrics
        print('Mean metrics')
        print(metrics_df.mean())
        print(f'iou_mean: {metrics_df.iloc[:, 2:].mean().mean():.4f}')
            
        return metrics_df
    
    def run_inference(self, loader, data_mode):
        """
        Make predictions for all batches in loader.
        """

        # List of dictionaries for predictions and metrics for the whole dataset in loader.
        predictions_list = []
        metrics_list = []

        # To log fancy progress bar
        loop = tqdm(loader)
        loop.set_description(f"Testing on {data_mode} set")

        for images, targets in loop:

            # Send images and targets to same device as model
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Inference step over single batch
            predictions_list_batch, metrics_list_batch = self.run_inference_batch(images, targets)

            # Update full output
            predictions_list += predictions_list_batch
            metrics_list += metrics_list_batch

            # Update progress bar
            loop.update()

        return predictions_list, metrics_list

    def run_inference_batch(self, images, targets):
        """
        Make predictions for each batch of images, and select the masks, and boxes with the best scores for each class 
        and image. The predicted quantities are then compared with the targets to extract the mask and box IoU 
        for each class.

        Returns:
            predictions_list: list of dictionaries of the form {masks, boxes} for each image
            metrics_list: list of dictionaries of the form {scores, iou_masks, iou_boxes} for each image.
        """

        # List of dictionaries for predictions and metrics for the whole batch.
        predictions_list = []
        metrics_list = []

        # Inference step
        with torch.no_grad():
            predictions = self.model(images)

        for target, pred in zip(targets, predictions):

            # Get labels and scores, which are aligned
            labels = pred['labels'].tolist()
            scores = pred['scores'].tolist()
            assert len(labels) == len(scores)

            # Get boxes and masks, which are also naturally aligned
            boxes = pred['boxes']
            masks = pred['masks']
            assert boxes.shape[0] == masks.shape[0] == len(labels)

            # Keep track of best-score mask and boxes predictions for each image
            pred_dict = {'masks': torch.zeros_like(target['masks']),
                        'boxes': torch.zeros_like(target['boxes'])}

            # Keep track of best scores, and average IoUs for each image
            metrics_dict = {'scores': np.zeros(self.n_classes),
                            'iou_masks': np.zeros(self.n_classes),
                            'iou_boxes': np.zeros(self.n_classes)}

            # Loop over classes (i.e. kit and membrane)
            for i, cls in enumerate(self.classes):
                if self.class_ids[i] in labels:  # Check whether there is at least one prediction for that class

                    # Get the maximum confidence class location (i.e. first occurrence in list)
                    class_loc = labels.index(self.class_ids[i])

                    # Get best class score
                    class_score = scores[class_loc]

                    # Get best class boxes and masks
                    class_box = boxes[class_loc]
                    class_mask = masks[class_loc, 0]

                    # Binarize masks
                    class_mask = (class_mask >= self.evaluation_args.mask_thresholds[i]).to(torch.uint8)

                    # Compute IoU
                    class_iou_mask = compute_iou_mask(class_mask, target['masks'][i])
                    class_iou_box = compute_iou_box(class_box, target['boxes'][i])

                    # Update dictionaries
                    pred_dict['masks'][i] = class_mask
                    pred_dict['boxes'][i] = class_box

                    metrics_dict['scores'][i] = class_score
                    metrics_dict['iou_masks'][i] = class_iou_mask
                    metrics_dict['iou_boxes'][i] = class_iou_box

                else:  # If there is no prediction, we leave all zeros in the dictionary
                    print(f'{cls} is missing from the prediction!')

            # Send mask and boxes to cpu
            pred_dict['masks'] = pred_dict['masks'].to('cpu')
            pred_dict['boxes'] = pred_dict['boxes'].to('cpu')

            # Update lists
            predictions_list.append(pred_dict)
            metrics_list.append(metrics_dict)

        return predictions_list, metrics_list

    @staticmethod
    def get_metrics(predictions, image_names):
        """
        Return metrics in the form of a Panda's dataframe with image_names as indices.
        """

        metrics_dict = {}

        for key in ['scores', 'iou_masks', 'iou_boxes']:
            for i, cls in enumerate(['kit', 'membrane']):
                metrics_dict[f'{key}_{cls}'] = [pred[key][i] for pred in predictions]

        metrics_df = pd.DataFrame.from_dict(metrics_dict)
        metrics_df.index = pd.Index(image_names, name='image_names')

        return metrics_df

    def save_results(self, save_path, metrics_df, kit_id, data_mode):
      """
      Save metrics_df as a csv file
      """

      # Create folder if it doesn't exist
      if not os.path.exists(save_path):
          Path(save_path).mkdir(parents=True, exist_ok=True)
      
      # Convert kit_id list into concatenated strings
      kit_id = '-'.join(kit_id)
      # Create the paths and ask whether to overwrite them or not
      stats_path = os.path.join(save_path, f'{kit_id}_{data_mode}_stats.csv')
      iou_mean_path = os.path.join(save_path, f'{kit_id}_{data_mode}_iou_mean.txt')
      info_path = os.path.join(save_path, f'{kit_id}_{data_mode}_info.yaml')
      
      # Compute statistics
      metrics_stats = metrics_df.describe().loc[['mean', 'min', 'max']]
      iou_mean = metrics_df.iloc[:, 2:].mean().mean()
      count = len(metrics_df)

      for _path in [stats_path, iou_mean_path, info_path]:
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
          if _path == stats_path:
              metrics_stats.to_csv(_path)
          elif _path == iou_mean_path:
              with open(_path, 'w') as f:
                f.write(str(iou_mean))
          elif _path == info_path:
              with open(_path, 'w') as f:
                  yaml.dump({'n_set': count}, f)
              
