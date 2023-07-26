from datasets import load_dataset
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import TrainingArguments
from datasets import Audio
import numpy as np

class Wav2vecModel():
    def __init__(self):
        pass
    def load_dataset(self):
        gtzan = load_dataset("marsyas/gtzan","all")
        gtzan = gtzan["train"].train_test_split(seed=42,shuffle=True,test_size=0.1)
        id2label_fn = gtzan["train"].features["genre"].int2str
        print(id2label_fn(gtzan["train"][0]["genre"]))
        return id2label_fn,gtzan

    def load_model(self):
        model_id = "ntu-spml/distilhubert"
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, do_normalize=True, return_attention_mask=True
        )
        sampling_rate = feature_extractor.sampling_rate
        id2label_fn,gtzan=self.load_dataset()
        gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
        sample = gtzan["train"][0]["audio"]
        print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        print(f"inputs keys: {list(inputs.keys())}")
        print(
            f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
        )
        return model_id,feature_extractor,gtzan

    def preprocess_function(self,examples):
        max_duration = 30.0
        model_id, feature_extractor, gtzan= self.load_model()
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )

        return inputs

    def map_function(self):
        model_id, feature_extractor, gtzan = self.load_model()
        id2label_fn, _ = self.load_dataset()
        gtzan_encoded = gtzan.map(
            self.preprocess_function, remove_columns=["audio", "file"], batched=True, num_proc=1)
        gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
        id2label = {
            str(i): id2label_fn(i)
            for i in range(len(gtzan_encoded["train"].features["label"].names))
        }
        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(id2label)
        model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label)


if __name__ == "__main__":

    wav = Wav2vecModel()
    wav.map_function()

