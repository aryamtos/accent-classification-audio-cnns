from prepare_dataset import LoadAndPreprocessToModel
from model import Conv1d1Lstm
from spectrogram import Spectrogram


if __name__ == "__main__":
    MAX_WAVE_SIZE = 195000 
    path_dir = 'BA/'
    load_preprocess = LoadAndPreprocessToModel(path_dir)
    filenames,labels = load_preprocess.shuffle_dataset()
    labels, encoder = load_preprocess.encoder_labels(labels)
    spec = Spectrogram(MAX_WAVE_SIZE)
    input_shape, preprocessing_fn= spec.process_spectrogram(filenames)
    print(input_shape)
    #print(len(encoder.categories_[0]))
    #model = spec.model_conv1dlstm('Conv1d1Lstm', input_shape, len(encoder.categories_[0]))
    

