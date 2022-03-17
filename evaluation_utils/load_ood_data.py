import logging

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from datasets.celebA_dataset import get_celebA_dataset, get_celebA_ood_dataset
from datasets.waterbirds_dataset import get_waterbird_dataloader, get_waterbird_dataset
from datasets.color_mnist import get_biased_mnist_dataloader, get_biased_mnist_dataset
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


known_dataset_sizes = {
  
  'cmnist' : (28,28),
  'waterbirds' : (224,224),
  'celebA' : (224,224)
}
def get_normalize_params(args):
    if args.model_arch == "DeiT":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else :
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    return mean, std

def get_resolution_from_dataset(dataset):
  if dataset not in known_dataset_sizes:
    raise ValueError(f"Unsupported dataset {dataset}.")
  return get_resolution(known_dataset_sizes[dataset])

def get_ood_loader(args):
   
    mean, std = get_normalize_params(args)
    if args.model_arch == "BiT":
        precrop, crop = get_resolution_from_dataset(args.dataset)
        transform_test = transforms.Compose([
                    transforms.Resize((crop, crop)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
    else :
        transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.id_dataset == "celebA":
        idtestset = get_celebA_dataset(split = "test", transform = transform_test, 
                                root_dir = 'datasets')
        oodtestset = get_celebA_ood_dataset(transform = transform_test, 
                               root_dir = 'datasets')
    
    if args.id_dataset == "waterbirds":
        idtestset = get_waterbird_dataset(data_label_correlation = 0.95, 
                        split="test", transform = transform_test, root_dir = 'datasets')

        oodtestset = torchvision.datasets.ImageFolder("datasets/ood_data/placesbg", transform=transform_test)
    
    if args.id_dataset == "cmnist":
    
        idtestset = get_biased_mnist_dataset(root = './datasets/MNIST', 
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=False, partial=True, cmap = "1",transform = transform_test)
        oodtestset = torchvision.datasets.ImageFolder("datasets/ood_data/partial_color_mnist_0&1", transform=transform_test)


    testloaderIn = torch.utils.data.DataLoader(idtestset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True) if idtestset is not None else None
    testloaderOut = torch.utils.data.DataLoader(oodtestset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True) if oodtestset is not None else None
    
    return testloaderIn, testloaderOut