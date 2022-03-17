"""
Color MNIST Dataset. Adapted from https://github.com/clovaai/rebias
"""
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
import warnings
import cv2
warnings.filterwarnings(action="ignore")
torch.manual_seed(3407)
np.random.seed(777)

class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP1 = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                 [255, 0, 0], [255, 0, 0],[255, 0, 0], [255, 0, 0], [255, 0, 0]]
    
    COLOUR_MAP2 = [[128, 0, 255], [255, 0, 128], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [255, 0, 0], [255, 0, 0],[0, 255, 0], [0, 255, 0], [0, 255, 0]] 


    def __init__(self, root, cmap, fg_color, bg_color, orig, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, partial=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cmap = cmap
        self.random = True
        self.Partial = partial
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.orig = orig
        #print(dir(self))
        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]
        # y, bg, cmap
        self.env_dict = {
            (0, 0, "1"): 0, #red 0 
            (0, 1, "1"): 1, #green 0 
            (1, 0, "1"): 2, #red 1  
            (1, 1, "1"): 3, #green 1 
            (0, 0, "2"): 4, #purple 0
            (0, 1, "2"): 5, #magenta 0
            (1, 0, "2"): 6, #purple 1
            (1, 1, "2"): 7  #magenta 1
        }
    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label, cmap):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)
        indices = np.where((self.targets == label).numpy())[0] 
        self._shuffle(indices);
        indices = torch.LongTensor(indices) 

        n_samples = len(indices)    
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))
        correlated_indices = indices[:n_correlated_samples]; 
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class);
        if self.Partial:
            other_labels = [_label % 2 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        else:
            other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
       
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices): 
            _label = other_labels[idx] 
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        if self.Partial: 
            n_labels = 2
        else: 
            n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)} 
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label) 

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label, self.cmap) 
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets]) 
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        bg_label = int(self.biased_targets[index])
        return img, target, self.env_dict[(target, bg_label, self.cmap)]


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, cmap, fg_color, bg_color,orig, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, partial = False):
        super(ColourBiasedMNIST, self).__init__(root,orig =orig, fg_color = fg_color, bg_color=bg_color, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels,
                                                partial = partial,
                                                cmap = cmap)
        
    def _binary_to_colour(self, data, bg_colour, fg_colour):
     
        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 1
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=3) 
        fg_data = fg_data * torch.ByteTensor(fg_colour)
        fg_data = fg_data.permute(0, 3, 1, 2)
        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0

        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3) 
        bg_data = bg_data * torch.ByteTensor(bg_colour)   
        bg_data = bg_data.permute(0, 3, 1, 2)
        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)

    def _make_biased_mnist(self, indices, label, cmap):
        if cmap == "1": 
          
            if self.orig == True :
                label = self.COLOUR_MAP1[label]
                fg = [255,255,255]
            else :
                label = self.bg_color
                fg = self.fg_color
                
        elif cmap == "2":
            if self.orig == True :
                label = self.COLOUR_MAP2[label]
                fg = [255,255,255]
            else :
                label = self.bg_color
                fg = self.fg_color
                
        return self._binary_to_colour(self.data[indices], label, fg), self.targets[indices]


def get_biased_mnist_dataloader( root, transform, batch_size, data_label_correlation, cmap,
                                n_confusing_labels=9, train=True, partial=False, 
                                orig = False, bg_color=[255,0,0], fg_color=[255,255,255]):
    kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}
    transform = transform
    dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation*2,
                                n_confusing_labels=n_confusing_labels, partial=partial, cmap = cmap, 
                                bg_color = bg_color, fg_color = fg_color, orig = orig)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return dataloader
