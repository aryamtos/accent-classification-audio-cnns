from prepare_dataset import LoadAndPreprocessToModel
from model import Conv1d1Lstm
from dataloader import SpectrogramDataset
from spectrogram import Spectrogram
from training import TrainLoaderDataset
import sys
import os
import argparse

def train(args):
    dataset_train = args.dataset_train
    dataset_val = args.dataset_val
    max_wave_size = args.max_wave_size
    noise_value = args.noise_value
    patience = args.patience
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    weight_decay_ = args.weight_decay_


    load_preprocess = LoadAndPreprocessToModel(dataset_train)
    filenames,labels_ = load_preprocess.shuffle_dataset()
    labels, encoder = load_preprocess.encoder_labels(labels_)

    load_preprocess = LoadAndPreprocessToModel(dataset_val)
    filenames_val, labels_val_ = load_preprocess.shuffle_dataset()
    labels_val, encoder = load_preprocess.encoder_labels(labels_val_)

    spec = Spectrogram(max_wave_size)
    input_shape, preprocessing_fn = spec.waveform_spectrogram(filenames[0])
    print(input_shape)
    model = spec.model_conv1dlstm(input_shape, 2)
    spectro = SpectrogramDataset(filenames, labels_, spec, noise_value)
    spectro_val = SpectrogramDataset(filenames_val,labels_val_,spec,noise_value)

    train = TrainLoaderDataset(spectro,spectro_val)
    #train_loader,valloader=train.loader()
    print(spectro.__len__())
    model, train_loss = train.train_model_early_stopping(model,patience,n_epochs,learning_rate,weight_decay_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Example of parser')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_train',type=str,required=True,help='Directory of dataset')
    parser_train.add_argument('--dataset_val',type=str,required=True,help='Directory of validation.')
    parser_train.add_argument('--max_wave_size',type=int,required=True)
    parser_train.add_argument('--noise_value',type=float,required=True)
    parser_train.add_argument('--patience',type=int,help='Description of patience argument')
    parser_train.add_argument('--n_epochs',type=int,help='Description of number of epochs')
    parser_train.add_argument('--learning_rate',type=float,help='Description of learning rate value')
    parser_train.add_argument('--weight_decay_',type=float,help='Description of decay value')


    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')

