import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SpectrogramDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.classes = sorted(os.listdir(folder_path))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        self.transform = transform

        for cls in self.classes:
            class_folder = os.path.join(folder_path, cls)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls])
    def convert_tensor(self, tensor):
        numpy_array = tensor.numpy()
        pytorch_tensor = torch.from_numpy(numpy_array)
        return pytorch_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        #image = np.array(image)
        #image = self.convert_tensor(image)
        #image = torch.from_numpy(image)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        #image = np.expand_dims(image, axis=0)
        #image = image[:, :97, :]
        #image = torch.from_numpy(image)

        label = self.labels[idx]
        return image,label
