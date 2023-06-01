FROM ubuntu:20.04

WORKDIR /accent-classification-audio

COPY teste  /accent-classification-audio/teste/
COPY requirements.txt ./requirements.txt

COPY . /accent-classification-audio

RUN apt-get update && apt-get install -y python3-pip 

RUN apt-get --yes install libsndfile1

RUN pip3 install -r requirements.txt

ENV PATH_DIR /accent-classification-audio/teste/

CMD ["python3","main.py"]