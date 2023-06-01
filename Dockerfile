FROM ubuntu:20.04

WORKDIR /accent-classification-audio

COPY requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y python3-pip 

RUN  apt-get install libsndfile1


RUN pip3 install -r requirements.txt

COPY . /accent-classification-audio

CMD ["python3","main.py"]