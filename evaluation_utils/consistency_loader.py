import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ConsistencyDataset(Dataset):
    def __init__(self, transform, root_dir):
        self.envs = ['land','water']
        self.root_dir  = root_dir
        self.dataset_name = [f'waterbird_{env}' for env in self.envs]
        self.dataset_dir = [os.path.join(self.root_dir, dataset_name,'images') for dataset_name in self.dataset_name]
        if not os.path.exists(self.dataset_dir[0]):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.') 
        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = transform
     
    def __len__(self):
        return len(self.filename_array)
    
    def __getitem__(self, idx):
        out_img = []
        y = self.y_array[idx]
        place = self.place_array[idx]
        for env_dir in self.dataset_dir:
            img_filename = os.path.join(
                env_dir,
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            img = self.transform(img)
            out_img.append(img)

        return out_img, y

def get_consistency_dataset(transform, root_dir):
   
    dataset = ConsistencyDataset(root_dir=root_dir, transform=transform)
    
    return dataset



