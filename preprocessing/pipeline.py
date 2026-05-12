"""
Preprocessing pipeline — frame extraction, face detection, dataset class.
"""

import os
import cv2
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN

IMAGE_SIZE = 380
FRAMES_PER_VIDEO = 10

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DeepfakeDualDataset(Dataset):
    def __init__(self, root_dir, transform_rgb):
        self.transform_rgb = transform_rgb
        self.samples = []
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.jpg')]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.jpg')]
        min_len = min(len(real_files), len(fake_files))
        if min_len == 0:
            raise RuntimeError(f"No data found in {root_dir}")
        self.samples = [(f, 0) for f in random.sample(real_files, min_len)]
        self.samples += [(f, 1) for f in random.sample(fake_files, min_len)]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform_rgb(Image.open(path).convert('RGB'))
        return img, img, torch.tensor(label).long()
