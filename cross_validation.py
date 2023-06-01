import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from pythorchtools import EarlyStopping



class CrossValidation:


    def __init__(self,k_folds,
                 num_epochs,
                 loss_function,
                 trainloader,
                 val_loader):
        pass

