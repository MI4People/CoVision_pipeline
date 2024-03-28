"""
File containing the TrainerClassification class, the main class for training and validating the classification model.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import copy

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from dataset_cla import MembraneZonesDataset
from dataloader_cla import init_dataloader
from model_cla import ClassificationModel
from transformations_cla import PreprocessMembraneZones, AugmentedMembraneZones
from utils_cla.miscellaneous import set_seed


class TrainerClassification:
    def __init__(self, args, parameters, do_validation=True):

        # Assign attributes
        self.args = args
        self.parameters = argparse.Namespace(**parameters)
        self.do_validation = do_validation
        self.kit_id = self.parameters.kit_id
        self.val_set = self.parameters.val_set
        self.shots = self.parameters.shots

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters.device = self.device
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss': [], 'accuracy': []}
        self.metrics_val = {'loss': [], 'accuracy': []} if self.do_validation else None

        # Set seed for reproducibility
        set_seed(self.parameters.seed)

        # Transformations
        self.transformation_train = AugmentedMembraneZones()
        self.transformation_val = PreprocessMembraneZones()

        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloaders()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)
        self.n_valset = len(self.loader_val.dataset)*self.loader_train.dataset.n_zones

        # Model architecture
        self.model = ClassificationModel().to(self.device)
        self.best_model = copy.deepcopy(self.model)  # Keep track of best model
        # Optimizer
        self.optimizer = optim.Adam(
            params=self.model.parameters(), 
            lr=float(self.parameters.lr), 
            weight_decay=float(self.parameters.weight_decay))
        # Criterion
        self.criterion = nn.BCELoss(reduction='mean')
        # Scheduler
        if self.parameters.scheduler_step is not None:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.parameters.scheduler_step, 
                gamma=self.parameters.scheduler_gamma)
        else:
            self.scheduler = None
        
        # Timestamp to identify training runs
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  
        self.file_path = os.path.join(self.args.save_path, time_str)
        
        # To continue training from checkpoint
        self.start_epoch = 0
        self.elapsed_time = timedelta(0)
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def train(self, save_state_bool=False, save_model_bool=False):
        """
        Trains and validates the model for all epochs, store the training and
        validation metrics, log results and save the best models.
        """        

        # To log fancy progress bar
        epoch_loop = tqdm(range(1 + self.start_epoch, self.parameters.epochs + 1))
        epoch_loop.set_description(f"Training for {self.parameters.epochs} epochs")  # Description

        start = time() - self.elapsed_time.seconds  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train, update train metric and return training loss
            epoch_train_loss, epoch_train_accuracy = self.train_epoch(epoch_ndx)
            
            if self.do_validation:
            
                # Validate, and return epoch loss and accuracy
                epoch_val_loss, epoch_val_accuracy = self.evaluate(self.loader_val, self.n_val, epoch_ndx)
                # Update metrics
                self.metrics_val['loss'].append(epoch_val_loss)                                                                                                                                                                                                                                 
                self.metrics_val['accuracy'].append(epoch_val_accuracy)

                # Update best model
                if epoch_val_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_val_accuracy
                    self.best_epoch = epoch_ndx
                    self.best_model = copy.deepcopy(self.model)
                    # Save model state
                    if save_model_bool:
                        self.save_model()
                               
                # Number of missclassified examples
                n_wrong = self.n_valset - int(epoch_val_accuracy*self.n_valset)
                # Update progress bar
                epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}, Val Acc = {(epoch_val_accuracy*100):.3f}, Ratio = {n_wrong}/{self.n_valset}")

                # Calculate elapsed time
                end = time()
                self.elapsed_time = timedelta(seconds=int(end-start))
            
                # Save full class state (in separate file if it is the best)
                if save_state_bool:
                    if epoch_val_accuracy == self.best_accuracy:
                        self.save_state(epoch_ndx, stamp='_best')
                    else:
                        self.save_state(epoch_ndx, stamp='')
                

            else:
                epoch_loop.set_postfix_str(f"Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}")             

            # Update scheduler and log the change in learning rate
            if self.scheduler is not None:
                before_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                after_lr = self.optimizer.param_groups[0]["lr"]
                if after_lr != before_lr:
                    print(f"Epoch {epoch_ndx}: lr {before_lr} -> {after_lr}")

    def init_dataloaders(self):
        """
        Initialize train and validation dataloaders.
        Batch_size is divided by the number of zones because of the way in which the dataset is retrieved
        and iterated over.
        """

        print("Loading data...")

        loader_train = init_dataloader(
            self.args, self.kit_id, 
            data_mode='train',
            n_batches=self.parameters.batch_size//2,
            n_workers=self.parameters.num_workers,
            shuffle=True,
            shots=self.shots, 
            transform=self.transformation_train)
               
        # Validation dataloader using 'val' or 'test' sets (as specified by self.val_set)
        if self.do_validation:
            loader_val = init_dataloader(
                self.args, self.kit_id, 
                data_mode=self.val_set,
                n_batches=self.parameters.batch_size//2,
                n_workers=self.parameters.num_workers,
                shuffle=False,
                shots=[None]*len(self.shots), 
                transform=self.transformation_val)
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
    def evaluate(self, loader, n_loader, epoch_ndx=None):
        """
        Evaluate the model over an entire dataloader and calculate performance metrics
        """

        # Set model to evaluation mode
        self.model.eval()

        # Store epoch loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # To log fancy progress bar
        loop = tqdm(loader, total=n_loader)
        if epoch_ndx is not None:
            loop.set_description(f"Epoch {epoch_ndx} | Evaluation")
        else:
            loop.set_description(f"Evaluation")

        for x, y in loop:
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
            loop.set_postfix_str(f"Loss = {loss:.4f} | Accuracy = {(accuracy*100):.3f}%")
        
        
        running_loss /= n_loader
        running_accuracy /= n_loader

        return running_loss, running_accuracy

    @staticmethod
    def get_accuracy(y_pred, y):
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

    @property
    def get_parameters(self):
        """
        Return parameters attribute but in dictionary format
        """
        return {k:v for k, v in self.parameters._get_kwargs()}

    def save_state(self, epoch_ndx, stamp):
        """
        Save whole instance state: model state, parameters, metrics, etc
        """
        state = {'args': self.args,
                 'parameters': {k:v for k, v in self.parameters._get_kwargs()},
                 'do_validation': self.do_validation,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
                 'metrics_train': self.metrics_train,
                 'metrics_val': self.metrics_val, 
                 'file_path': self.file_path,
                 'epoch': epoch_ndx,
                 'elapsed_time': self.elapsed_time,
                 'best_accuracy': self.best_accuracy,
                 'best_model_state': self.best_model.state_dict()}
        torch.save(state, self.file_path + stamp + '.state')

    def save_model(self):
        """
        Save only model state 
        """
        torch.save({'model_state': self.model.state_dict()}, self.file_path + '_model.state')

    @classmethod
    def from_saved_state(cls, state_path, new_epochs=None):
        """
        Initialize Trainer class from previous saved state. This is useful, for instance, when training 
        was interrupted and we want to resume from where it stopped. 
        """

        print("Loading saved state...")
        
        state = torch.load(state_path, map_location=torch.device('cpu'))
        parameters = state['parameters']
        # Update new epochs in parameters dictionary
        if new_epochs is not None:
            if new_epochs <= parameters['epochs']:
              raise ValueError(f"new_epochs ({new_epochs}) must be bigger than current epochs ({parameters['epochs']})!")
            parameters['epochs'] = new_epochs
        
        print(f"Epoch: {state['epoch']}")
        print(f"Best accuracy: {state['best_accuracy']}")
        print(f"Elapsed time: {state['elapsed_time']}")

        trainer = cls(args=state['args'], 
                      parameters=parameters,
                      do_validation=state['do_validation'])
        
        # Load model, optimizer and scheduler states
        trainer.model.load_state_dict(state['model_state'])
        trainer.best_model.load_state_dict(state['best_model_state'])
        trainer.optimizer.load_state_dict(state['optimizer_state'])
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(state['scheduler_state'])
        # Load metrics
        trainer.metrics_train = state['metrics_train']
        trainer.metrics_val = state['metrics_val']
        # To continue training from checkpoint correctly
        trainer.best_accuracy = state['best_accuracy']
        trainer.start_epoch = state['epoch']
        trainer.elapsed_time = state['elapsed_time']
        trainer.file_path = state['file_path']

        return trainer

