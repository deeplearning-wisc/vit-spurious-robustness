
from __future__ import print_function
from evaluation_utils.calMetric import get_and_print_results
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import time


def get_energy_score(inputs, model):
    T = 1.0 # Temperature constant
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        outputs = model(inputs)
    scores = -to_np((T*torch.logsumexp(outputs / T, dim=1)))
    return -scores

def testData(args, testloaderIn, testloaderOut, model):
    
    model.eval()
    t0 = time.time()
    g1 = open(f"energy_results/{args.name}/{args.id_dataset}/{args.model_arch}/{args.model_type}_In.txt", 'w')
    g2 = open(f"energy_results/{args.name}/{args.id_dataset}/{args.model_arch}/{args.model_type}_Out.txt", 'w')
    N = len(testloaderIn.dataset)
    id_scores = []; ood_scores = []
    count = 0
    print("Processing in-distribution images")
########################################In-distribution###########################################
    for j, data in enumerate(testloaderIn):
        images, _,_ = data
        inputs = images.cuda()
        inputs.requires_grad = True
        scores = get_energy_score(inputs,model)
        id_scores.append(scores)
        count += images.shape[0]
        for score in scores :
            g1.write("{}\n".format(score))

        print("{:4}/{:4} images processed, {:.4f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()
    t0 = time.time()
    N = len(testloaderOut.dataset)
    count = 0
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloaderOut):
        images, _ = data
        inputs = images.cuda()
        inputs.requires_grad=True
        count += images.shape[0]
        scores = get_energy_score(inputs,model)
        ood_scores.append(scores)
        for score in scores:
            g2.write("{}\n".format(score))
        print("{:4}/{:4} images processed, {:.4f} seconds used.".format(count, N, time.time()-t0))
        t0 = time.time()
        
    print(f"ID Dataset: {args.id_dataset}")
    print(f"Model Type : {args.model_arch}")
    get_and_print_results(ood_scores, id_scores)


