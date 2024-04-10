
import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
from pytorchtools import EarlyStopping
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

class TrainLoaderDataset:

    def __init__(self,spectro_data,spectro_val):
        self.spectro_data = spectro_data
        self.spectro_val = spectro_val

    def compute_weights(self,spectro_data):
        labels_tensor = torch.from_numpy(spectro_data.labels)
        class_counts = torch.bincount(labels_tensor)
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        return weights


    def loader(self):
        weights_train = self.compute_weights(self.spectro_data)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights_train, len(self.spectro_data), replacement=True)
        trainloader = torch.utils.data.DataLoader(self.spectro_data, batch_size=64, sampler=weighted_sampler)
        valloader = torch.utils.data.DataLoader(self.spectro_val,batch_size=64,num_workers=0)

        return trainloader,valloader

    def train_model_early_stopping(self,model,patience,n_epochs,learning_rate,weight_decay_):
        criterion = nn.CrossEntropyLoss()
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
        train_losses = []
        avg_train_losses = []

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        trainloader,valloader= self.loader()
        writer = SummaryWriter()

        for epoch in range(1, n_epochs + 1):
            model.train()
            for batch, (data, target) in enumerate(trainloader, 1):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            epoch_len = len(str(n_epochs))
            print_msg = f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] train_loss: {train_loss:.5f} '
            print(print_msg)
            writer.add_scalar('Loss/Train', train_loss, epoch)

            train_losses = []

            early_stopping(train_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load('checkpoint.pt'))
        writer.close()

        return model, avg_train_losses







