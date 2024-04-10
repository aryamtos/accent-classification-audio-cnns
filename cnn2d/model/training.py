from tensorboardX import SummaryWriter
import pytorchtools
from pytorchtools import EarlyStopping
import numpy as np
import torch
import torch.nn as nn

def train_model_early_stopping(device,trainloader,valloader,model, batch_size, 
                                patience,n_epochs,optimizer,scheduler,criterion):

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    writer = SummaryWriter()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        model.train()
        for batch, (data, target) in enumerate(trainloader, 1):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        for data, target in valloader:
            #with torch.no_grad():
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_losses.append(loss.item())

        scheduler.step()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', valid_loss, epoch)
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses