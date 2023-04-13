from prepare_dataset import LoadAndPreprocessToModel
from model import Conv1d1Lstm
import librosa
import torch
import torchaudio
import wave
import soundfile as sf
import numpy as np


class Spectrogram:

    def __init__(self,MAX_WAVE_SIZE):
        self.MAX_WAVE_SIZE = MAX_WAVE_SIZE

    def load_audio_extract_waveform(self, file_path):
        waveform,__ = torchaudio.load(file_path)
        if waveform.shape[0] < self.MAX_WAVE_SIZE:
            zero_padding = torch.zeros(self.MAX_WAVE_SIZE - waveform.shape[0], dtype=torch.float32)
            waveform = torch.cat([waveform.unsqueeze(0), zero_padding.unsqueeze(1)], 0)

        waveform = waveform[:self.MAX_WAVE_SIZE]
        return waveform 

    def process_spectrogram(self,waveform):

        stft = torch.stft(waveform, n_fft=3000, hop_length=2000, window=torch.hann_window(3000))
        spectrogram = torch.abs(stft)
        return spectrogram.numpy()

    def waveform_spectrogram(self, sample):
        spectro_fn = lambda f: self.process_spectrogram(self.load_audio_extract_waveform(f))
        spectro_shape = spectro_fn(sample).shape
        return spectro_shape, spectro_fn
    

    def model_conv1dlstm(self,input_shape,num_labels):

            return Conv1d1Lstm(input_shape,num_labels)
        
    
    



    



