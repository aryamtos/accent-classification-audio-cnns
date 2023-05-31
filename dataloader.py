import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
from torch.utils.data import random_split
from torch.utils.data import Subset
from sklearn.model_selection import KFold 

class SpectrogramDataset(Dataset):
    def __init__(self, filenames, labels, spec):
        self.filenames = filenames
        self.labels = labels
        self.spec = spec
        self.input_shape, self.preprocessing_fn= spec.waveform_spectrogram(filenames[0])

    def convert_tensor(self, tensor):
        numpy_array = tensor.numpy()
        pytorch_tensor = torch.from_numpy(numpy_array)
        return pytorch_tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        spectrogram = self.preprocessing_fn(filename)
        spectrogram = self.convert_tensor(spectrogram)
        #spectrogram = spectrogram.transpose(0,1)  # altera a ordem das dimensões
        label = torch.tensor(label, dtype=torch.long)
        
        return spectrogram, label
    

class SplitAndComputeWeights():

    def __init__(self,spectro):
        self.spectro
  
    
    def __len__(self):
        return len(self.spectro)

    def split_train(self):
        numb_items = self.__len__()
        numb_train = round(numb_items * 0.7)
        numb_val = round(numb_items * 0.15)
        numb_test = numb_items - numb_train - numb_val
        print(numb_train, numb_val, numb_test)
        train_ds, val_ds, test_ds = random_split(self.spectro,[numb_train,numb_val,numb_test])
        return train_ds, val_ds, test_ds
    
    def compute_weights(self,train_ds):
        labels_tensor = torch.from_numpy(train_ds.dataset.labels)
        class_counts = torch.bincount(labels_tensor)
        weights = 1.0/ class_counts.float()
        weights = weights / weights.sum()

        return weights
    
    def distribute_weights(self):

        train_ds, val_ds, test_ds = self.split_train()
        weights_train = self.compute_weights(train_ds)
        weights_val = self.compute_weights(val_ds)
        weighted_sampler_train =  torch.utils.data.WeightedRandomSampler(weights_train, len(train_ds), replacement=True)
        weighted_sampler_val = torch.utils.data.WeightedRandomSampler(weights_val, len(val_ds), replacement=True)
        trainloader = torch.utils.data.DataLoader(train_ds, batch_size=32, sampler= weighted_sampler_train)
        val_dataloader = torch.utils.data.DataLoader(val_ds,batch_size=32, sampler=weighted_sampler_val)
        test_dataloader = torch.utils.data.DataLoader(test_ds,batch_size=32, num_workers=0)

        return trainloader, val_dataloader,test_dataloader
    

    
        


