from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter
from evaluation_utils.transform_utils import get_normalize_params, get_resolution_from_dataset
from datasets.color_mnist import get_biased_mnist_dataloader
import models.bits as bits
import timm
import os

model_dict = {'ViT-B_16':'vit_base_patch16_224_in21k', 
'ViT-S_16':'vit_small_patch16_224_in21k',
'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
'DeiT-B_16':'deit_base_patch16_224', 
'DeiT-S_16':'deit_small_patch16_224',
'DeiT-Ti_16':'deit_tiny_patch16_224'}

np.random.seed(777)
BG_COLOR_MAP = [[240, 96, 7], [236, 240, 43], [15, 245, 241],[87, 49, 21],[133, 125, 15],
             [1, 92, 36],[171, 0, 103],[251, 183, 250], [209, 237, 149],[0, 38, 255]]
FG_COLOR_MAP = [[0,0,0],[255,255,255]]

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def calculate_consistency(args):
    mean, std = get_normalize_params(args)
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(args.output_dir,args.name, args.dataset, args.model_arch, args.model_type)
    if args.model_arch ==  "ViT" or args.model_arch == "DeiT":
            model = timm.create_model(
                    model_dict[args.model_type],
                    pretrained=False,
                    num_classes=2,
                )
            model.load_state_dict(torch.load(args.checkpoint_dir + ".bin"))
            model.eval()
            transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                 transforms.Normalize(mean, std),])
           
    if args.model_arch == "BiT":

                model = bits.KNOWN_MODELS[args.model_type](head_size=2, zero_head=False)
                model = torch.nn.DataParallel(model)
                checkpoint = torch.load(args.checkpoint_dir + ".pth.tar", map_location="cpu")
                model.load_state_dict(checkpoint["model"])
                transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                                 transforms.Normalize(mean,std),])
    try :
            if torch.cuda.is_available():
                model = model.cuda()
    except Exception:
            raise Exception("No CUDA enabled device found. Please Check !")
  
    if args.setup == "Random":
        FG_COLOR = FG_COLOR_MAP[np.random.choice(2)]
        BG_COLOR = BG_COLOR_MAP[np.random.choice(len(BG_COLOR_MAP))]

    elif args.setup == "BW":
        FG_COLOR = [0,0,0] #Black
        BG_COLOR = [255,255,255] #White
    else:
        raise Exception("Unknown Setup")
    
    testloader = get_biased_mnist_dataloader(root = '~/Documents/CMNIST/datasets/MNIST', transform=transform, batch_size=batch_size,
                                                data_label_correlation= 0.45,
                                                n_confusing_labels= 1,
                                                train=False, partial=True, cmap = "1", orig= False,fg_color=FG_COLOR, bg_color=BG_COLOR)

    def accuracy(out, label):
            _,pred= torch.max(out,dim=1);
            return torch.tensor(torch.sum(pred==label).item()/len(pred))
        

    acc = []
    
    for j, data in enumerate(testloader):
            images, labels,_ = data;
            inputs = images.cuda()

            inputs.requires_grad=True
            out = model(inputs);
            acc.append(accuracy(out, labels.cuda()));
    acc = np.array(acc)
    return np.mean(acc)

if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CMNIST Consistency Results')
    parser.add_argument("--name", required=True,
                        help="help identify checkpoint")
    parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                        default="ViT",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint_dir",
                        help="directory of saved model checkpoint")
    parser.add_argument("--model_type", required= True, default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The directory where checkpoints are stored.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--setup",default="Random", required =True, choices =["Random","BW"],type = str, 
                        help = "Setup for calculating CMNIST consistency" )
    
    args = parser.parse_args()

     
    random_runs = 50
    acc = []
    for _ in np.arange(random_runs):
        acc.append(calculate_consistency(args))
    print(f"Consistency of {args.model_type} is {sum(acc)/ len(acc)}")