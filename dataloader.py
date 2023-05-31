import torch
from torch.utils.data import Dataset,DataLoader, ConcatDataset, WeightedRandomSampler
import torchaudio
import torch.optim as optim

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



if __name__ == "__main__":

    spectro_spotify = SpectrogramDataset(filenames_spotify,labels_spotify,preprocessing_fn_spotify)

