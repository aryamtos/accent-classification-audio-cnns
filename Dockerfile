FROM python:3.10

WORKDIR /accent-classification-audio

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3-pip 
RUN apt-get --yes install libsndfile1
RUN pip install -r requirements.txt

COPY . .

ENV DATASET_TRAIN="dataset/train/"
ENV DATASET_VAL="dataset/validation/"
ENV MAX_WAVE_SIZE=195000
ENV NOISE_VALUE=0.1
ENV PATIENCE=25
ENV N_EPOCHS=50
ENV LR=0.0001
ENV DECAY=0.001

CMD ["python", "main.py", "train", "--dataset_train=$DATASET_TRAIN", "--dataset_val=$DATASET_VAL", "--max_wave_size=$MAX_WAVE_SIZE", "--noise_value=$NOISE_VALUE", "--patience=$PATIENCE", "--n_epochs=$N_EPOCHS", "--learning_rate=$LR", "--weight_decay_=$DECAY"]
