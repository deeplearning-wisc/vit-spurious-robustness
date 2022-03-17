from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
from utils.data_utils import get_loader_inference
from collections import defaultdict, Counter
import models.bits as bits 
import timm
import logging
import os

logger = logging.getLogger(__name__)


model_dict = {'ViT-B_16':'vit_base_patch16_224_in21k', 
'ViT-S_16':'vit_small_patch16_224_in21k',
'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
'DeiT-B_16':'deit_base_patch16_224', 
'DeiT-S_16':'deit_small_patch16_224',
'DeiT-Ti_16':'deit_tiny_patch16_224'}

class Result:

        def __init__(self):
            self.acc = defaultdict(list)
            self.counter_env = Counter()
            self.counter_label = Counter()
        
        def update(self, out, env, label):
            self.counter_env.update(env)
            self.counter_label.update(label)
            self.accuracy_per_env( out, env, label)

        def accuracy_per_env(self, out, env, label):
            for env_unique in np.unique(env):
                ind = env==env_unique
                acc_env = accuracy(out[ind], label[ind])
                self.acc[env_unique].append(acc_env)

        def output_result(self):
            print("The number of examples according to environment :")
            for key, val in self.counter_env.items():
                print(f"{key} : {val}", end = '  ')
            
            print("\n **************** ")
            print("The number of examples according to labels :")
            for key, val in self.counter_label.items():
                print(f"{key} : {val}", end = '  ')

            print("\n **************** ")
            print("\nThe accuracy according to environemnt :")
            for key, val in self.acc.items():
                acc = sum(val)/len(val)
                print(f"{key} : {acc}", end = '  ')
            
            print("\n")

def accuracy(out, label):
                pred= np.argmax(out,axis=1); 
                return np.sum(pred==label)/len(pred)

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

def get_acc(loader, model):
    result = Result()
    acc = []
    count = 0
    for j, data in enumerate(loader):
            images, labels, env = data
            count+=len(images)
            inputs = images.cuda()
            inputs.requires_grad=True
            out = model(inputs);
            acc.append(accuracy(out.detach().cpu().numpy(), labels.cpu().numpy()))
            result.update(out.detach().cpu().numpy(),env.cpu().numpy(), labels.cpu().numpy())
    return result, acc

def calculate_acc(args):
    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(args.output_dir,args.name, args.dataset, args.model_arch, args.model_type)
    if args.model_arch ==  "ViT" or args.model_arch == "DeiT":
            model = timm.create_model(
                    model_dict[args.model_type],
                    pretrained=False,
                    num_classes=2,
                    img_size = args.img_size
                )
            
            model.load_state_dict(torch.load(args.checkpoint_dir + ".bin"))
            model.eval()

    if args.model_arch == "BiT":

                model = bits.KNOWN_MODELS[args.model_type](head_size=2, zero_head=False)
                model = torch.nn.DataParallel(model)
                checkpoint = torch.load(args.checkpoint_dir + ".pth.tar", map_location="cpu")
                model.load_state_dict(checkpoint["model"])
                
    try :
            if torch.cuda.is_available():
                model = model.cuda()
    except Exception:
            raise Exception("No CUDA enabled device found. Please Check !")
  
    logger.info(f"Inference for Dataset: {args.dataset} \t Model : {args.model_type} ")
    trainloader, testloader = get_loader_inference(args)
    logger.info("Calculating Accuracy Metrics on Train data")
    result_train, acc_train = get_acc(trainloader, model)
    logger.info(f"Average Train Accuracy = {np.mean(np.array(acc_train))}")
    result_train.output_result()

    logger.info("Calculating Accuracy Metrics on Test data")
    result_test, acc_test = get_acc(testloader, model)
    logger.info(f"Average Test Accuracy = {np.mean(np.array(acc_test))}")
    result_test.output_result()


if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Accuracy Metrics')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='batch size (default: 64) used for inference')
    parser.add_argument('--split', default='train', type=str,
                    help='Split used for inference (train or test)')
    parser.add_argument('--model_arch', default='ViT', type=str,
                    help='Model Architecture (ViT or BiT or DeiT)')
    parser.add_argument('--model_type', default='ViT-S_16', type=str,
                    help='Corresponding model version')
    parser.add_argument('--checkpoint', required=True, type=str,
                    help='Saved Model Checkpoint')
    args = parser.parse_args()

    calculate_acc(args) 