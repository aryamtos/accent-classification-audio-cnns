FROM python:3.10

WORKDIR /accent-classification-audio

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3-pip 
RUN apt-get --yes install libsndfile1
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "train", "--dataset_train=dataset/train/", "--dataset_val=dataset/validation/", "--max_wave_size=195000", "--noise_value=0.1", "--patience=25", "--n_epochs=50", "--learning_rate=0.0001", "--weight_decay_=0.001"]
