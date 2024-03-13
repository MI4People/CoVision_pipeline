"""
File containing the TrainingSegmentation class, the main class for training and validating the segmentation model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from dataset_classification import FullMembraneDataset
from model_classification import ClassificationModel
from utils_classification.utils import set_seed

class TrainerClassification:
    def __init__(self, args, transformation_train=None, transformation_val=None):

        # Assign attributes
        self.args = args
        self.transformation_train = transformation_train
        self.transformation_val = transformation_val

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.device = self.device
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss': [], 'accuracy': []}
        self.metrics_val = {'loss': [], 'accuracy': []} if self.args.do_validation else None

        # Set seed for reproducibility
        set_seed(self.args.seed)

        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloader()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)
        self.n_valset = len(self.loader_val.dataset)

        # Model architecture, criterion, optimizer and scheduler
        self.model = ClassificationModel().to(self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=float(self.args.lr), weight_decay=float(self.args.weight_decay))
        self.criterion = nn.BCELoss(reduction='mean')
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # Unique instance identifier
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.stamp = time_str
        self.file_path = f'{self.args.saved_models_dir}/{self.stamp}'
        # To keep track of best model
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def train(self, save_bool=False):
        """
        Trains and validates the model for all epochs, store the training and
        validation metrics, log results and save the best models.
        """        

        # To log fancy progress bar
        epoch_loop = tqdm(range(1, self.args.epochs + 1), total=self.args.epochs)
        epoch_loop.set_description(f"Training for {self.args.epochs} epochs")  # Description

        start = time()  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train, update train metric and return training loss
            epoch_train_loss, epoch_train_accuracy = self.train_epoch(epoch_ndx)
            
            if self.args.do_validation:
            
                # Validate, update validation metrics and return epoch loss and accuracy
                epoch_val_loss, epoch_val_accuracy = self.validate(epoch_ndx)
                
                if epoch_val_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_val_accuracy
                    self.best_epoch = epoch_ndx
                
                # Number of missclassified examples
                n_wrong = int((1 - epoch_val_accuracy)*self.n_valset)
                # Update progress bar
                epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}, Val Acc = {(epoch_val_accuracy*100):.3f}, Ratio = {n_wrong}/{self.n_valset}")
            else:
                epoch_loop.set_postfix_str(f"Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}")
                
            # Update scheduler and log the change in learning rate
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            if after_lr != before_lr:
                print(f"Epoch {epoch_ndx}: lr {before_lr} -> {after_lr}")
            
            # Save model (in separate file if it is the best)
            if save_bool:
                end = time()
                elapsed_time = str(timedelta(seconds=int(end-start)))
                if self.args.do_validation and epoch_val_accuracy == self.best_accuracy:
                    self.save_model(epoch_ndx, stamp='_best', elapsed_time=elapsed_time)
                else:
                    self.save_model(epoch_ndx, stamp='', elapsed_time=elapsed_time)


    def init_dataloader(self):
        """
        Initialize train and validation dataloaders
        """

        print("Loading data...")

        dataset_train = FullMembraneDataset(self.args, set_mode='train', transform=self.transformation_train)
        loader_train = DataLoader(dataset=dataset_train,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  num_workers=self.args.num_workers,
                                  pin_memory=True)
                                  
        if self.args.do_validation:

            dataset_val = FullMembraneDataset(self.args, set_mode='test', transform=self.transformation_val)
            loader_val = DataLoader(dataset=dataset_val, 
                                    batch_size=self.args.batch_size, 
                                    shuffle=False, 
                                    num_workers=self.args.num_workers, 
                                    pin_memory=True)
        else:
            loader_val = []

        return loader_train, loader_val


    def train_epoch(self, epoch_ndx):
        """
        Train model for one epoch and update training metrics
        """

        # To store running train loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # To log fancy progress bar
        train_loop = tqdm(self.loader_train, total=self.n_train)
        train_loop.set_description(f"Epoch {epoch_ndx} | Training")

        self.model.train()
        for x, y in train_loop:

            # Send to device
            x = x.to(self.device)
            y = y.to(self.device)
            # Set gradients to zero
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x)
            # Compute loss
            loss = self.criterion(y_pred, y)
            # Backward pass
            loss.backward()
            # Optimize
            self.optimizer.step()
            # Update running loss
            running_loss += loss.item()
            # Train accuracy
            with torch.no_grad():
                accuracy = self.get_accuracy(y_pred, y).item()
                running_accuracy += accuracy

            # Update progress bar
            train_loop.set_postfix_str(f"Loss = {loss.item():.4f} | Accuracy = {(accuracy*100):.3f}%")

        # Update train metric
        running_loss /= self.n_train
        running_accuracy /= self.n_train
        self.metrics_train['loss'].append(running_loss)
        self.metrics_train['accuracy'].append(running_accuracy)

        return running_loss, running_accuracy

    @torch.no_grad()
    def validate(self, epoch_ndx):
        """
        Validate the model over the entire validation set, calculate and update performance metrics
        """

        # Set model to evaluation mode
        self.model.eval()

        # Store epoch loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # To log fancy progress bar
        val_loop = tqdm(self.loader_val, total=self.n_val)
        val_loop.set_description(f"Epoch {epoch_ndx} | Validation")

        for x, y in val_loop:
            # Send to device
            x = x.to(self.device)
            y = y.to(self.device)
            # Forward pass
            y_pred = self.model(x)
            # Compute loss
            loss = self.criterion(y_pred, y).item()
            # Compute accuracy
            accuracy = self.get_accuracy(y_pred, y).item()

            # Update running loss and accuracy
            running_loss += loss
            running_accuracy += accuracy

            # Update progress bar
            val_loop.set_postfix_str(f"Loss = {loss:.4f} | Accuracy = {(accuracy*100):.3f}%")
        
        # Update metrics
        running_loss /= self.n_val
        running_accuracy /= self.n_val
        self.metrics_val['loss'].append(running_loss)
        self.metrics_val['accuracy'].append(running_accuracy)

        return running_loss, running_accuracy

    def get_accuracy(self, y_pred, y):
        acc = np.mean((y.squeeze() == (y_pred.squeeze() > 0.5)).to(int).tolist())
        return acc
        

    def get_metrics(self):
        """
        Returns metrics as Pandas DataFrames ready to be plotted
        """

        metrics_val_df = pd.DataFrame.from_dict(self.metrics_val)
        metrics_val_df.index.name = 'epochs'

        metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
        metrics_train_df.index.name = 'epochs'

        return metrics_train_df, metrics_val_df

    def save_model(self, epoch_ndx, stamp, elapsed_time):
        """
        Save model state, parameters, and metrics.
        """
        
        state = {
            'model_state': self.model.state_dict(),  # model's state
            'args': self.args,
            'metrics_train': self.metrics_train,
            'metrics_val': self.metrics_val,
            'stamp': self.stamp,
            'epoch_ndx': epoch_ndx,
            'best_accuracy': self.best_accuracy,
            'elapsed_time': elapsed_time
        }
        torch.save(state, self.file_path + stamp + '.state')

    def save_txt_file(self):
        with open(self.file_path + '.txt', 'w') as file:
            file.write(str(self.args) + '\n\n')
            file.write(f'Best accuracy: {str(self.best_accuracy)}\n')
            file.write(f'Ratio: {str(int((1 - self.best_accuracy)*self.n_valset))}\n')
            file.write(f'Best epoch: {str(self.best_epoch)}\n')


    def load_state(self, path, device):
        """
        Load model state and metrics
        """
        state = torch.load(path, map_location=torch.device(device))
        # Load model's state and metrics
        self.model.load_state_dict(state['model_state'])
        self.metrics_train = state['metrics_train']
        self.metrics_val = state['metrics_val']
