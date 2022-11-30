import logging

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from datasets.celebA_dataset import get_celebA_dataloader, get_celebA_dataset
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

def get_loader_train(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    mean, std = get_normalize_params(args)
    print(mean,std)
    if args.model_arch == "BiT":
        precrop, crop = get_resolution_from_dataset(args.dataset)
        transform_train = transforms.Compose([
                transforms.Resize((precrop, precrop)),
                transforms.RandomCrop((crop, crop)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
         ])
        transform_test = transforms.Compose([
                    transforms.Resize((crop, crop)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
    else :
        transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
        transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.dataset == "celebA":
        trainset = get_celebA_dataset(split = "train", transform = transform_train, 
                                root_dir = 'datasets')
        testset = get_celebA_dataset(split = "val", transform = transform_test, 
                               root_dir = 'datasets')
    
    if args.dataset == "waterbirds":
        trainset = get_waterbird_dataset(data_label_correlation = 0.95, 
                        split="train", transform = transform_train, root_dir = 'datasets')

        testset = get_waterbird_dataset(data_label_correlation = 0.95, 
                        split="val", transform = transform_test,root_dir = 'datasets')
    
    if args.dataset == "cmnist":
        trainset_1 = get_biased_mnist_dataset(root = './datasets/MNIST',
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "1",transform = transform_train)
        trainset_2 = get_biased_mnist_dataset(root = './datasets/MNIST', 
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "2",transform = transform_train)
        testset = get_biased_mnist_dataset(root = './datasets/MNIST', 
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=False, partial=True, cmap = "1",transform = transform_test)
        trainset = trainset_1 + trainset_2


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

def get_loader_inference(args):
   
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

    if args.dataset == "celebA":
        trainset = get_celebA_dataset(split = "train", transform = transform_test, 
                                root_dir = 'datasets')
        testset = get_celebA_dataset(split = "test", transform = transform_test, 
                               root_dir = 'datasets')
    
    if args.dataset == "waterbirds":
        trainset = get_waterbird_dataset(data_label_correlation = 0.95, 
                        split="train", transform = transform_test, root_dir = 'datasets')

        testset = get_waterbird_dataset(data_label_correlation = 0.95, 
                        split="test", transform = transform_test,root_dir = 'datasets')
    
    if args.dataset == "cmnist":
        trainset_1 = get_biased_mnist_dataset(root = './datasets/MNIST',
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "1",transform = transform_test)
        trainset_2 = get_biased_mnist_dataset(root = './datasets/MNIST', 
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "2",transform = transform_test)
        testset = get_biased_mnist_dataset(root = './datasets/MNIST', 
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=False, partial=True, cmap = "1",transform = transform_test)
        trainset = trainset_1 + trainset_2
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True) if testset is not None else None
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True) if testset is not None else None
    
    return train_loader, test_loader
