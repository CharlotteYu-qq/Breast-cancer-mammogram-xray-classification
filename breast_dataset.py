# dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pydicom
import numpy as np
import os
from typing import Optional, Tuple, Dict


class BreastMammoDataset(Dataset):
    def __init__(self, dataframe, transform: Optional[transforms.Compose] = None, 
                 image_size: Tuple[int, int] = (512, 512), 
                 is_train: bool = False): 
        
        self.dataframe = dataframe
        self.image_size = image_size
        self.is_train = is_train

        # label remapping
        self.label_map = {
            'BENIGN': 0,
            'MALIGNANT': 1,
            'BENIGN_WITHOUT_CALLBACK': 2
        }

            # define transforms
        if transform is None:
            if self.is_train:
                
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    # random horizontal flip
                    transforms.RandomHorizontalFlip(p=0.5), 
                    # random rotation
                    transforms.RandomRotation(degrees=10), 
                    # random affine
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    # color jitter
                    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                # validation/test transforms
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        dicom_path = row['image_path']
        label_str = row['pathology']

        # load DICOM image
        image = self.load_dicom_image(dicom_path)

        # convert to PIL grayscale image
        image_pil = Image.fromarray(image).convert('L')

        # apply transforms
        image_tensor = self.transform(image_pil)

        # ensure single channel
        if image_tensor.ndim == 3 and image_tensor.shape[0] != 1:
            image_tensor = image_tensor[0:1]
        elif image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # label conversion
        if label_str not in self.label_map:

            print(f"[error] unknown pathology label: {label_str} in {dicom_path}")
            label = -1 # unknown label
        else:
            label = self.label_map[label_str]

        return {'img': image_tensor, 'label': torch.tensor(label, dtype=torch.long)}

    def load_dicom_image(self, dicom_path):
        """load DICOM image and convert to numpy array"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array.astype(np.float32)

            # normalize to 0-255
            image_array -= np.min(image_array)
            image_array /= (np.max(image_array) + 1e-8)
            image_array *= 255.0

            return image_array.astype(np.uint8)

        except Exception as e:
            print(f"[warn] unable to load DICOM file: {os.path.basename(dicom_path)} | error: {e}")
            return np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
    
    
    