from prepare_dataset import LoadAndPreprocessToModel
from model import Conv1d1Lstm
import librosa
import torch
import torchaudio
import wave
import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio

class Spectrogram:

    def __init__(self,MAX_WAVE_SIZE):
        self.MAX_WAVE_SIZE = MAX_WAVE_SIZE

    def load_preprocess_coraa(self,file_path):
        audio_data, sample_rate = sf.read(file_path)
    
        waveform = tf.constant(audio_data, dtype=tf.float32)
    
        if tf.shape(waveform) < self.MAX_WAVE_SIZE:
            zero_padding = tf.zeros(self.MAX_WAVE_SIZE - tf.shape(waveform), dtype=tf.float32)
            waveform = tf.concat([waveform, zero_padding], 0)

        waveform = waveform[:self.MAX_WAVE_SIZE]
    
        return waveform.numpy()
   
    def compute_mel_spectrogram(waveform):
        mel_spectrogram = tfio.experimental.audio.melscale(waveform, rate=16000, mels=128, fmin=0, fmax=8000)  
        dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
        freq_mask = tfio.experimental.audio.freq_mask(dbscale_mel_spectrogram, param=10)
        return freq_mask
    
    def load_and_preprocess_waveform(self,file_path):
        audio_binary = tf.io.read_file(file_path)
        
        # Check the bit depth of the WAV file
        with wave.open(file_path, 'rb') as wav_file:
            bit_depth = wav_file.getsampwidth() * 8
            
        # If the bit depth is not 16, convert the file to 16-bit
        if bit_depth != 16:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, dtype='float32')
            y_16bit = (y * (2**15)).astype(np.int16)
            y_16bit_path = file_path + '.16bit.wav'
            sf.write(y_16bit_path, y_16bit, sr, subtype='PCM_16')
            audio_binary = tf.io.read_file(y_16bit_path)
            file_path = y_16bit_path
        
        # Decode the 16-bit WAV file
        waveform, _ = tf.audio.decode_wav(audio_binary)
        
        # Remove the temporary 16-bit WAV file
        if bit_depth != 16:
            os.remove(y_16bit_path)
            
        waveform = tf.squeeze(waveform, axis=-1)
        waveform = tf.cast(waveform, tf.float32)
        if tf.shape(waveform) < self.MAX_WAVE_SIZE:
            zero_padding = tf.zeros(self.MAX_WAVE_SIZE - tf.shape(waveform), dtype=tf.float32)
            waveform = tf.concat([waveform, zero_padding], 0)

        waveform = waveform[:self.MAX_WAVE_SIZE]
        return waveform.numpy()

    def process_spectrogram(self,waveform):
                #print(waveform.shape)
                spectrogram = tf.signal.stft(waveform, frame_length=3000, frame_step=2000)
                spectrogram = tf.abs(spectrogram)
                return spectrogram
        
    def waveform_spectrogram_spotify(self, sample):
                spectro_fn = lambda f: self.compute_mel_spectrogram(self.load_and_preprocess_waveform(f))
                spectro_shape = spectro_fn(sample).shape
                return spectro_shape, spectro_fn
    
    def waveform_spectrogram_coraa(self, sample):
                spectro_fn = lambda f: self.compute_mel_spectrogram(self.load_preprocess_coraa(f))
                spectro_shape = spectro_fn(sample).shape
                return spectro_shape, spectro_fn
    def model_conv1dlstm(self,input_shape,num_labels):

            return Conv1d1Lstm(input_shape,num_labels)
        
    
    



    



