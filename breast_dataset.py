# 1Nov dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pydicom
import numpy as np
import os
from typing import Optional, Tuple, Dict
import random


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
                self.transform = self.get_medical_transforms(image_size)
            else:
                # validation/test transforms
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        else:
            self.transform = transform

    def get_medical_transforms(self, image_size=(512, 512)):
        """medical image specific augmentations"""
        return transforms.Compose([
            transforms.Resize(image_size),

            # Geometric transformations - maintain medical validity
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # Reduce rotation angle

            # Elastic transformations - simulate breast tissue deformation
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),  # Reduce translation magnitude
                scale=(0.98, 1.02),      # Reduce scaling range
                shear=2                  # Small shear
            ),

            # Intensity transformations - simulate different exposure conditions
            transforms.ColorJitter(
                brightness=0.05,    # Reduce brightness variation
                contrast=0.05,      # Reduce contrast variation
            ),

            # Medical image specific augmentations
            transforms.Lambda(self.medical_specific_augmentation),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def medical_specific_augmentation(self, img):
        """Medical image specific augmentation methods"""
        import numpy as np
        from PIL import Image

        # Randomly choose a medical augmentation (30% chance to apply)
        if random.random() < 0.3:
            aug_type = random.choice(['gaussian_noise', 'gamma_correction', 'local_contrast'])
            
            if aug_type == 'gaussian_noise':
                # Add slight Gaussian noise - simulate image noise
                img_array = np.array(img)
                noise = np.random.normal(0, 2, img_array.shape).astype(np.uint8)  # Reduce noise intensity
                img_array = np.clip(img_array + noise, 0, 255)
                return Image.fromarray(img_array)
            
            elif aug_type == 'gamma_correction':
                # Gamma correction - simulate different contrast
                gamma = random.uniform(0.95, 1.05)  # Reduce gamma range
                img_array = np.array(img).astype(np.float32)
                img_array = 255 * (img_array / 255) ** gamma
                return Image.fromarray(img_array.astype(np.uint8))
            
            elif aug_type == 'local_contrast':
                # Local contrast enhancement
                img_array = np.array(img).astype(np.float32)

                # Randomly choose a local region to enhance contrast
                h, w = img_array.shape
                patch_h, patch_w = h // 4, w // 4
                start_h = random.randint(0, h - patch_h)
                start_w = random.randint(0, w - patch_w)
                
                patch = img_array[start_h:start_h+patch_h, start_w:start_w+patch_w]
                patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8) * 255
                img_array[start_h:start_h+patch_h, start_w:start_w+patch_w] = patch
                
                return Image.fromarray(img_array.astype(np.uint8))
        
        return img

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

            # Medical image window level adjustment (conservative version)
            image_array = self.apply_conservative_window_level(image_array, dicom_data)

            # normalize to 0-255
            image_array -= np.min(image_array)
            image_array /= (np.max(image_array) + 1e-8)
            image_array *= 255.0

            return image_array.astype(np.uint8)

        except Exception as e:
            print(f"[warn] unable to load DICOM file: {os.path.basename(dicom_path)} | error: {e}")
            return np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
    
    def apply_conservative_window_level(self, image_array, dicom_data):
        """Conservative window level adjustment"""
        try:
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth

                # Handle multi-value cases
                if hasattr(window_center, '__iter__'):
                    window_center = window_center[0]
                if hasattr(window_width, '__iter__'):
                    window_width = window_width[0]

                # Use a narrower window (conservative adjustment)
                window_min = window_center - window_width // 3  # Originally // 2
                window_max = window_center + window_width // 3
                
                image_array = np.clip(image_array, window_min, window_max)
                
        except Exception as e:
            # If window level adjustment fails, keep original
            pass
            
        return image_array








# # dataset.py

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms
# import pydicom
# import numpy as np
# import os
# from typing import Optional, Tuple, Dict


# class BreastMammoDataset(Dataset):
#     def __init__(self, dataframe, transform: Optional[transforms.Compose] = None, 
#                  image_size: Tuple[int, int] = (512, 512), 
#                  is_train: bool = False): 
        
#         self.dataframe = dataframe
#         self.image_size = image_size
#         self.is_train = is_train

#         # label remapping
#         self.label_map = {
#             'BENIGN': 0,
#             'MALIGNANT': 1,
#             'BENIGN_WITHOUT_CALLBACK': 2
#         }

#             # define transforms
#         if transform is None:
#             if self.is_train:
                
#                 self.transform = transforms.Compose([
#                     transforms.Resize(image_size),
#                     # random horizontal flip
#                     transforms.RandomHorizontalFlip(p=0.5), 
#                     # random rotation
#                     transforms.RandomRotation(degrees=10), 
#                     # random affine
#                     transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
#                     # color jitter
#                     transforms.ColorJitter(brightness=0.1, contrast=0.1), 
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5], std=[0.5])
#                 ])
#             else:
#                 # validation/test transforms
#                 self.transform = transforms.Compose([
#                     transforms.Resize(image_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5], std=[0.5])
#                 ])
#         else:
#             self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         dicom_path = row['image_path']
#         label_str = row['pathology']

#         # load DICOM image
#         image = self.load_dicom_image(dicom_path)

#         # convert to PIL grayscale image
#         image_pil = Image.fromarray(image).convert('L')

#         # apply transforms
#         image_tensor = self.transform(image_pil)

#         # ensure single channel
#         if image_tensor.ndim == 3 and image_tensor.shape[0] != 1:
#             image_tensor = image_tensor[0:1]
#         elif image_tensor.ndim == 2:
#             image_tensor = image_tensor.unsqueeze(0)

#         # label conversion
#         if label_str not in self.label_map:

#             print(f"[error] unknown pathology label: {label_str} in {dicom_path}")
#             label = -1 # unknown label
#         else:
#             label = self.label_map[label_str]

#         return {'img': image_tensor, 'label': torch.tensor(label, dtype=torch.long)}

#     def load_dicom_image(self, dicom_path):
#         """load DICOM image and convert to numpy array"""
#         try:
#             dicom_data = pydicom.dcmread(dicom_path)
#             image_array = dicom_data.pixel_array.astype(np.float32)

#             # normalize to 0-255
#             image_array -= np.min(image_array)
#             image_array /= (np.max(image_array) + 1e-8)
#             image_array *= 255.0

#             return image_array.astype(np.uint8)

#         except Exception as e:
#             print(f"[warn] unable to load DICOM file: {os.path.basename(dicom_path)} | error: {e}")
#             return np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
    
    
    