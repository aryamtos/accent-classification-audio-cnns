import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold 
from pythorchtools import EarlyStopping


def train_model(model,batch_size,patience,n_epochs):

    train_losses=[]
    valid_losses=[]
    avg_train_losses=[]
    avg_valid_losses=[]

    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in range(1,n_epochs+1):

        model.train()
        