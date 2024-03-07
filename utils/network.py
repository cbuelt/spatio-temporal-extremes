#
# This file includes helper functions for training the neural network.
#

import numpy as np
import torch
from evaluation.metrics import extremal_coefficient

class Scheduler:
    """ Scheduler with early stopping and checkpoint saving.
    """
    def __init__(self, path, name,  patience = 3, min_delta = 0):
        self.output_path = path + name + ".pt"
        self.best_loss = np.inf
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, current_loss, epoch, model):
        # Early stopping and checkpointing
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            print(f"\nBest validation loss: {self.best_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), self.output_path)
        elif current_loss > (self.best_loss+self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping activated")
                return True
        return False
    
