import pandas as pd
import os
import time
from pydub import AudioSegment
import shutil
import re
import librosa
from sklearn.cluster import KMeans
import os
import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb")
from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

num_speakers = 2  # @param {type:"integer"}

language = 'any'  # @param ['any', 'English']

model_size = 'medium'  # @param ['tiny', 'base', 'small', 'medium', 'large']
model = whisper.load_model(model_size)
class DiarizationAudioFiles(object):

    def __init__(self, dir):
        self.dir = dir
        self.audio = Audio()

    def duration_context_audio(self, directory):
        with contextlib.closing(wave.opean(directory, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration

    def transcribe_segments_files(self):

        list_duration_audios = []
        list_segments = []
        list_audios = []
        for audios in os.listdir(self.dir):
            audio_file = os.path.join(self.dir, audios)
            duration = self.duration_context_audio(audio_file)
            result = model.transcribe(audio_file)
            segments = result['segments']
            print(segments)
            list_audios.append(audio_file)
            list_duration_audios.append(duration)
            list_segments.append(segments)

        return list_duration_audios, list_segments, list_audios

    def segment_embedding(self, segment, duration, audio_file):

        start = segment['start']
        end = min(duration, segment['end'])
        clip = Segment(start, end)
        waveform, sample_rate = self.audio.crop(audio_file, clip)
        return embedding_model(waveform[None])

    def add_speaker_name(self):
        embeddings = self.get_embeddings()
        _, list_segments, list_audios = self.transcribe_segments_files()
        num_speakers = 2
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for segments in list_segments:
            for i, segment in enumerate(segments):
                segment['speaker'] = 'SPEAKER' + str(labels[i] + 1)
        return list_segments

    def transcript_by_speaker_identification(self):
        list_segments = self.add_speaker_name()
        file = open("arquivo.txt", "w+", encoding='utf-8')
        for segments in list_segments:
            for i, segment in enumerate(segments):
                if i == 0 or segments[i - 1]['speaker'] != segment['speaker']:
                    file.write('\n' + segment["speaker"] + ' ' + time.strftime('%H:%M:%S',
                                                                               time.gmtime(segment["start"])) + '\n')
                file.write(segment["text"][1:] + ' ')
        file.close()

    def get_embeddings(self):
        list_duration_audios, list_segments, list_audios = self.transcribe_segments_files()
        embeddings = np.zeros(shape=(len(list_segments), 192))
        for i, (segment, duration, audio_file) in enumerate(zip(list_segments, list_duration_audios, list_audios)):
            embeddings[i] = self.segment_embedding(segment[i], duration, audio_file)
        embeddings = np.nan_to_num(embeddings)
        return embeddings


if __name__ == "__main__":

    dir = "teste/"
    diarization_audio = DiarizationAudioFiles(dir)
    print("Hello World")

    diarization_audio.transcript_by_speaker_identification()
    list_duration_audios, list_segments, list_audios = diarization_audio.transcribe_segments_files()

    len(list_segments)



