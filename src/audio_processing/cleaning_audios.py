
from pydub import AudioSegment
from spleeter.separator import Separator
import os
import shutil
from tqdm import tqdm

class CleaningAudiosDataset(object):

    def __init__(self,path_with_folders,save_local_folder,batch_size,destination_path):
        self.path_with_folders = path_with_folders
        self.save_local_folder = save_local_folder
        self.batch_size = batch_size
        self.destination_path = destination_path

    def separate_voice_from_music(self):
        
        audio_files = [os.path.join(self.path_with_folders, f) for f in os.listdir(self.path_with_folders) if f.endswith('.wav')]

        batches = [audio_files[i:i + self.batch_size] for i in range(0, len(audio_files), self.batch_size)]
        separator = Separator('spleeter:2stems') 
        
        for batch in tqdm(batches):
            for audio_file in batch:
                prediction = separator.separate_to_file(audio_file, self.save_local_folder)

    def rename_and_copy(self,path):
        for dirpath,dirname,filenames in os.walk(path):
            for i, filename in enumerate(filenames):
                audio_file = os.path.join(dirpath,filename)
                if  not (audio_file.endswith('accompaniment.wav')):
                    dir_name = os.path.basename(dirpath)
                    file_name = os.path.basename(filename)
                    new_file = os.path.join(dirpath,f"{dir_name}.wav")
                    #os.rename(audio_file, new_file)
                    new_path = os.path.join(self.destination_path, f"{dir_name}.wav")
                    shutil.move(audio_file, new_path)


    def remove_low_volume(self,threshold):
        for file in os.listdir(self.destination_path):
            if file.endswith('.wav'):
                sound = AudioSegment.from_file(os.path.join(self.destination_path,file),format='wav')
                loud = [seg for seg in sound if seg.dBFS > threshold]
                if loud: 
                    new = loud[0]
                    for l in loud[1:]:
                        new+=l

                    output = os.path.join(self.destination_path,file)
                    new.export(output,format='wav')
                else:
                    print("Low loud")
                
    def process_audio_files(self):
        self.separate_voice_from_music()
        self.rename_and_copy(self.save_local_folder)
        self.remove_low_volume(threshold=0)  # Define o threshold de volume desejado

    def main(self):
        self.process_audio_files()