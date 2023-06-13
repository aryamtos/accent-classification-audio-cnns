import os
from concurrent.futures import ThreadPoolExecutor
import csv
import pandas as pd
from pydub import AudioSegment

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid


seg = Segmenter()

class SegmentationAudio:


    def __init__(self,audio_directory,save_directory,time_seconds,sample_rate):

        self.audio_directory = audio_directory
        self.save_directory = save_directory
        self.time_seconds = time_seconds
        self.sample_rate = sample_rate
         
    def process_audio_file(self,file):

        segmentation = seg(file)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        rows = []
        for row in segmentation:
            res = (filename,) + tuple(row)
            # print(res)
            rows.append(res)

        return rows

    def process_audio_files(self,input_dir, output_file):
        with ThreadPoolExecutor() as executor:
            files = []
            for dirpath, dirnames, filenames in os.walk(input_dir):
                for filename in filenames:
                    if filename.endswith('.ogg'):
                        files.append(os.path.join(dirpath, filename))
            all_rows = []
            for rows in executor.map(self.process_audio_file, files):
                print(rows)
                all_rows.extend(rows)

            with open(output_file, 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename','genre','start_time','end_time'])
                writer.writerows(all_rows)


    def read_csv_info(self,file_csv,input_dir):
        list_all_genres=[]
        df = pd.read_csv(file_csv,delimiter=',')
        for index,row in df.iterrows():
             audio_filename = row['Filename']
             genre = row['genre']
             start_time=float(row['start_time'])
             end_time=float(row['end_time'])

             for dirpath,dirnames, filenames in os.walk(input_dir):
                for filename in filenames:
                     audio_path = os.path.join(dirpath,filename)
                     audio = AudioSegment.from_file(audio_path)
                     if genre in ['male','female']:
                         trimmed_audio = audio[start_time * 1000:end_time * 1000]
                         trimmed_filename = f"trimmed_{audio_filename}"
                         trimmed_filepath = os.path.join(dirpath, trimmed_filename)
                         trimmed_audio.export(trimmed_filepath, format='wav')
                         #list_all_genres.append(trimmed_audio)
                         print(f"SALVAR AQUI'{trimmed_filepath}'.")
    
        #combined_audio = AudioSegment.empty()
        #for audio_segment in list_all_genres:
              #combined_audio += audio_segment

        #combined_audio.export("combined_audio.wav", format="wav")
        #print("AQUI.")


    def trim_audio_files(self):
        with ThreadPoolExecutor() as executor:
            for dirpath, dirnames, filenames in os.walk(self.audio_directory):
                for i, filename in enumerate(filenames):
                    audio_path = os.path.join(dirpath,filename)
                    audio_ = AudioSegment.from_file(audio_path)
                    trimmed_audio = audio_[:self.time_seconds * 1000]
                    resample_audio = trimmed_audio.set_frame_rate(self.sample_rate)
                    mono_audio = resample_audio.set_channels(1)
                    output_trimmed_audios = os.path.join(self.save_directory,filename)
                    mono_audio.export(output_trimmed_audios + ".wav",format="wav") 

        
if __name__ == "__main__":
    input_dir = "podcast_audio_only_one/"
    output = 'my_csv_file.csv'
    audio_directory = ""
    save_directory = ""
    time_seconds = 10 
    sample_rate = 16000
    seg_ = SegmentationAudio(audio_directory,save_directory,time_seconds,sample_rate)
    seg_.trim_audio_files()
    #seg_.read_csv_info(output,input_dir)

    #process_audio_files(input_dir, output)
