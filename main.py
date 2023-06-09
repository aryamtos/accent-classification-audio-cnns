from prepare_dataset import LoadAndPreprocessToModel
from model import Conv1d1Lstm
from spectrogram import Spectrogram
from dataloader import SpectrogramDataset, SplitAndComputeWeights
from training import TrainingModel, EvaluationModel
import sys
import os


if __name__ == "__main__":
    MAX_WAVE_SIZE = 195000 
    path_dir = "teste/"
    load_preprocess = LoadAndPreprocessToModel(path_dir)
    filenames,labels_ = load_preprocess.shuffle_dataset()
    labels, encoder = load_preprocess.encoder_labels(labels_)
    spec = Spectrogram(MAX_WAVE_SIZE)
    input_shape, preprocessing_fn= spec.waveform_spectrogram(filenames[0])
    print(input_shape)
    model = spec.model_conv1dlstm(input_shape, len(encoder.categories_[0]))
    print(model)
    spectro = SpectrogramDataset(filenames,labels_,spec)
    split_train_compute_weights = SplitAndComputeWeights(spectro)
    train_ds, val_ds, test_ds = split_train_compute_weights.split_train()
    print(train_ds,val_ds,test_ds)
    trainloader, val_dataloader,test_dataloader=split_train_compute_weights.distribute_weights()
    
    tr_model = TrainingModel(model,trainloader,val_dataloader)
    model, train_loss, valid_loss = tr_model.train_model(model)
    eval_model = EvaluationModel(model,trainloader,val_dataloader,test_dataloader)
    eval_model.evaluation()
    eval_model.confusion_matrix_plot()


