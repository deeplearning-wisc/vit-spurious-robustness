from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
import os
from evaluation_utils.transform_utils import get_normalize_params, get_resolution_from_dataset
from evaluation_utils.consistency_loader  import get_consistent_dataset
from collections import defaultdict, Counter
import models.bits as bits
import timm

model_dict = {'ViT-B_16':'vit_base_patch16_224_in21k', 
'ViT-S_16':'vit_small_patch16_224_in21k',
'ViT-Ti_16':'vit_tiny_patch16_224_in21k',
'DeiT-B_16':'deit_base_patch16_224', 
'DeiT-S_16':'deit_small_patch16_224',
'DeiT-Ti_16':'deit_tiny_patch16_224'}

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
            transform = transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor(),
                                 transforms.Normalize(mean, std),])

    if args.model_arch == "BiT":

                model = bits.KNOWN_MODELS[args.model_type](head_size=2, zero_head=False)
                model = torch.nn.DataParallel(model)
                checkpoint = torch.load(args.checkpoint_dir + ".pth.tar", map_location="cpu")
                model.load_state_dict(checkpoint["model"])
                transform = transforms.Compose([transforms.Resize((480, 480)),transforms.ToTensor(),
                                 transforms.Normalize(mean, std),])
    try :
            if torch.cuda.is_available():
                model = model.cuda()
    except Exception:
            raise Exception("No CUDA enabled device found. Please Check !")
    testsetout = get_consistent_dataset(root_dir= 'datasets',transform=transform)
    testloader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,shuffle=True, num_workers=2)
    
    def accuracy(out, label):
            _,pred= torch.max(out,dim=1); 
            return torch.tensor(torch.sum(pred==label).item()/len(pred))
    
    orig_data_pred = []
    change_data_pred = []
    count = 0
    for j, data in enumerate(testloader):
            print(f"Batch Number : {j}")
            (images_orig,images_change), label = data
            count += len(images_orig)
            assert images_orig.shape == images_change.shape

            orig_output = model(images_orig.cuda());
            change_output = model(images_change.cuda());
            _,real_pred = torch.max(orig_output,dim = 1)
            _,fake_pred = torch.max(change_output,dim = 1)
            correct_pred_index = real_pred == label.cuda()
            orig_data_pred.extend(real_pred[correct_pred_index].detach().cpu().numpy())
            change_data_pred.extend(fake_pred[correct_pred_index].detach().cpu().numpy())
    bool_array = [int(orig_data_pred[i]==change_data_pred[i]) for i in range(len(orig_data_pred))]
    print(f"Consistency Measure for {args.model_type} on Waterbirds = {sum(bool_array)/count}")

if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="help identify checkpoint")
    parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                        default="ViT",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint_dir",
                        help="directory of saved model checkpoint")
    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The directory where checkpoints are stored.")
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    
    args = parser.parse_args()

    calculate_consistency(args) 