import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold 
from pythorchtools import EarlyStopping

class TrainingModel:

    def __init__(self,
                 trainloader,
                 val_dataloader,
                 optimizer,
                 criterion,
                 batch_size,
                 n_epochs,
                 patience,
                 loss_function):
        self.trainloader = trainloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 32
        self.n_epochs = 250
        self.patience = 150
        self.loss_function = nn.CrossEntropyLoss()
    def train_model(self,model):

        train_losses=[]
        valid_losses=[]
        avg_train_losses=[]
        avg_valid_losses=[]


        early_stopping = EarlyStopping(patience=self.patience,verbose=True)

        for epoch in range(1,self.n_epochs+1):

            model.train()
            for batch,(data,target) in enumerate(self.trainloader,1):

                self.optimizer.zero_grad()
                output=model(data)
                loss=self.criterion(output,target)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            model.eval()

            for data,target in self.val_dataloader:

                output = model(data)
                loss = self.criterion(output,target)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                         f'train_loss:{train_loss:.5f}'+
                         f'valid_loss:{valid_loss:.5f}')
            print(print_msg)

            train_losses = []
            valid_losses = []

            early_stopping(valid_loss,model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))

        return model, avg_train_losses,avg_valid_losses




        