import librosa
import torchaudio
import wave
import soundfile as sf
import numpy as np
import os

import glob
import pathlib
import wave
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import librosa
from sklearn.model_selection import train_test_split
import soundfile as sf
import os
import torch.nn as nn

import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold



class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(16,16)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(8,8)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(4,4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32,6),
            nn.Softmax(dim=1)
        )

        self.gradient = None

    def activations_hook(self,grad):
        self.gradient = grad

    def forward(self,images):

        x = self.feature_extractor(images)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)

        return x

    def get_activation_gradients(self):
        return self.gradient

    def get_activation(self,x):
        return self.feature_extractor(x)

        