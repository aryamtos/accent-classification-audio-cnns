import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold 
from pytorchtools import EarlyStopping 
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import seaborn as sns


#Pytorchtools based on : Bjarten repository-https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class TrainingModel:

    def __init__(self,
                 model,
                 trainloader,
                 val_dataloader):
        self.trainloader = trainloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 32
        self.n_epochs = 50
        self.patience = 10
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

            print_msg = (f'[{epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}]' +
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


class EvaluationModel(TrainingModel):
    def __init__(self,
                 model,
                 trainloader,
                 val_dataloader,testloader):
         super().__init__(model,trainloader,val_dataloader)
         self.testloader = testloader
         self.class_correct = list(0. for i in range(10))
         self.class_total = list(0. for i in range(10))
         self.model = model
         self.criterion = nn.CrossEntropyLoss()
         self.batch_size = 64
         self.model_eval = self.model.eval()
         self.test_loss = 0.0

    def evaluation(self):

          for data,target in self.testloader:
                if len(target.data) != self.batch_size:
                    break

                output = self.model(data)
                loss = self.criterion(output, target)
                self.test_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                correct = np.squeeze(pred.eq(target.data.view_as(pred)))
                for i in range(self.batch_size):
                    label = target.data[i]
                    self.class_correct[label] += correct[i].item()
                    self.class_total[label] += 1

          self.test_loss = self.test_loss/len(self.testloader.dataset)
          print('Test Loss: {:.6f}\n'.format(self.test_loss))


    def confusion_matrix_plot(self):

          y_pred = []
          y_true = []

          for inputs,labels in self.testloader:
              output =  self.model(inputs)
              output = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
              y_pred.extend(output)
              labels = labels.data.numpy()
              y_true.extend(labels)

          classes =('AM','BA')
          cf_matrix = confusion_matrix(y_true, y_pred)
          df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],columns=[i for i in classes])
          plt.figure(figsize=(12, 7))
          sns.heatmap(df_cm, annot=True)
          plt.savefig('output.png')


    def plot_chart_metrics(self):

        precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)

        for i in range(len(precision)):
            print(f"Class {i}:")
            print(f"Precision: {precision[i]}")
            print(f"Recall: {recall[i]}")
            print(f"F1-score: {f1_score[i]}")
            print(f"Support: {support[i]}")

        plt.figure(figsize=(10, 5))
        sns.barplot(x=np.arange(len(precision)), y=precision)
        plt.title('Precision por classe')
        plt.xlabel('Classe')
        plt.ylabel('Precision')
        plt.xticks(np.arange(len(precision)), np.arange(len(precision)))
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.barplot(x=np.arange(len(recall)), y=recall)
        plt.title('Recall por classe')
        plt.xlabel('Classe')
        plt.ylabel('Recall')
        plt.xticks(np.arange(len(recall)), np.arange(len(recall)))
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.barplot(x=np.arange(len(f1_score)), y=f1_score)
        plt.title('F1-score por classe')
        plt.xlabel('Classe')
        plt.ylabel('F1-score')
        plt.xticks(np.arange(len(f1_score)), np.arange(len(f1_score)))
        plt.show()













        