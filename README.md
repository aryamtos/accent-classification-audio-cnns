# Accent Classification - Brazillian Portuguese

# **Installation**

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
```

### Container

```bash
docker exec -it accentBr /bin/bash
```

