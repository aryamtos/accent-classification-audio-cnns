# Accent Classification Brazillian Portuguese using CNN models

This repository showcases a classification task for Brazilian Portuguese using two model configurations: a 1D Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM), and a standalone 2D CNN. The CNN1D + LSTM model, based on <a href="https://github.com/wagnertostes/Classificacao-de-Sotaques-Brasileiros-usando-Redes-Neurais-Profundas/">Tostes'  work</a>, utilizes a <b>range of frequency values</b> from a spectrogram as input. Meanwhile, the CNN2D model processes images sized at 227x227 pixels.


### Installation

- **Local :**

```bash
git clone https://github.com/aryamtos/accent-classification-audio.git
pip3 install -r requirements.txt
```

- **Conda Environment** 🐍

```bash
git clone https://github.com/aryamtos/accent-classification-audio.git
conda create --name myenv
conda install --file requirements.txt
conda list
```

### Build Docker image 🐳

```bash
docker build -t accent:2.0 .
docker images
```

### Run Container Docker 

```bash
docker run -it --gpus all -v vol/:/vol/ --name accentBr -d accent:2.0
docker exec -it accentBr /bin/bash
```


