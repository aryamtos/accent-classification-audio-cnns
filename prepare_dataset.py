import glob
import pathlib
import wave
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import librosa
from sklearn.model_selection import train_test_split
import soundfile as sf
import os


class LoadAndPreprocessToModel():

    def __init__(self, directory):
        self.directory = directory

    def verify_directory_exists(self):
   
        if not os.path.exists(self.directory):
            info = f'PATH {self.directory} NOT FOUND'
            raise Exception(info)

    def get_lists_filenames_and_labels(self):
        list_filenames = []
        list_labels = []
        self.verify_directory_exists()
        data_dir = pathlib.Path(self.directory)
        classes = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
        print(f'Classes:{classes}')

        for index, name in enumerate(classes):
            class_dir = data_dir/ name
            filenames = sorted([str(item) for item in class_dir.glob('*') if item.is_file() and item.suffix == '.wav'
                                and os.path.getsize(item)!=44])
            labels = np.array([index]*len(filenames),dtype=np.int32)

            list_filenames.extend(filenames)
            list_labels.extend(labels)

        indices = np.arange(len(list_filenames))
        np.random.shuffle(indices)
        list_filenames = np.array(list_filenames)[indices]
        list_labels = np.array(list_labels)[indices]
        return list_filenames,list_labels
    

    def shuffle_dataset(self):
        list_filenames,list_labels = self.get_lists_filenames_and_labels()
        indices = np.arange(len(list_filenames))
        np.random.shuffle(indices)
        filenames = list_filenames[indices]
        labels = list_labels[indices]
        return filenames, labels
    
    def encoder_labels(self,labels):

        labels = labels.reshape((len(labels),1))
        enc = OneHotEncoder(sparse=False)
        onehot = enc.fit_transform(labels)
        return onehot, enc
    
    
    



    
        


        