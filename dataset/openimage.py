import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetJson(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        json_path = os.path.join(data_path, 'train_files.json') # data/dataset/openimages/train_files.json
        assert os.path.exists(json_path), f"please first run: python3 openimage_json.py"
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def build_openimage(args, transform):
    return DatasetJson(args.data_path, transform=transform)



class DatasetJson_2(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        json_path_OpenI = ""
        json_path_In1k = ""
        
        self.transform = transform
        
        assert os.path.exists(json_path_OpenI), f"please first run: python3 openimage_json.py"
        assert os.path.exists(json_path_In1k), f"please first run: python3 imagenet_json.py"
        
        with open(json_path_OpenI, 'r') as f:
            self.image_paths = json.load(f)
        
        with open(json_path_In1k, 'r') as f:
            self.image_paths += json.load(f)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')
    
    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)

def build_t2i_openi_IN1k(args, transform):
    return DatasetJson_2(args.data_path, transform=transform)
