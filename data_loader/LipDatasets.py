import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys
sys.path.append("../")

from data_loader.utils import get_category, to_bgr, random_choice, safe_crop 

train_images_id = 'data/instance-level_human_parsing/Training/train_id.txt'
train_images_folder = 'data/instance-level_human_parsing/Training/Images'
train_categories_folder = 'data/instance-level_human_parsing/Training/train_segmentations'
valid_images_id = 'data/instance-level_human_parsing/Validation/val_id.txt'
valid_images_folder = 'data/instance-level_human_parsing/Validation/Images'
valid_categories_folder = 'data/instance-level_human_parsing/Validation/Category_ids'

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class LIPDataset(Dataset):
    def __init__(self, split, num_classes, transformer):
        self.usage = split

        if split == 'train':
            self.id_file = train_images_id
            self.images_folder = train_images_folder
            self.categories_folder = train_categories_folder
        else:
            self.id_file = valid_images_id
            self.images_folder = valid_images_folder
            self.categories_folder = valid_categories_folder

        with open(self.id_file, 'r') as f:
            self.names = f.read().splitlines()

        self.transformer = transformer
        self.num_classes = num_classes
    def __getitem__(self, i):
        name = self.names[i]
        filename = os.path.join(self.images_folder, name + '.jpg')
        img = cv.imread(filename)
        image_size = img.shape[:2]
        category = get_category(self.categories_folder, name)

        x, y = random_choice(image_size)
        img = safe_crop(img, x, y)
        category = safe_crop(category, x, y)
        category = np.clip(category, 0, self.num_classes - 1)

        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            category = np.fliplr(category)

        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        y = category

        return img, torch.from_numpy(y.copy())

    def __len__(self):
        return len(self.names)

if __name__ == "__main__":
    dataset = LIPDataset('train')
    print(dataset[0])
