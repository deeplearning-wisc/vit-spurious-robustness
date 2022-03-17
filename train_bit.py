import logging
from os.path import join as pjoin  
import time
import os
import numpy as np
import torch
import torchvision as tv
import utils.lbtoolbox as lb
import models.bits as models
import utils.bit_common as bit_common
import utils.bit_hyperrule as bit_hyperrule
from itertools import cycle
from utils.data_utils import get_loader_train


def accuracy(out, label):
            _,pred= torch.max(out,dim=1);
            return torch.tensor(torch.sum(pred==label).item()/len(pred))

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
      for i in iterable:
        yield i


def run_eval(model, data_loader, device, chrono, logger, step):
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1= [], []
  end = time.time()
  for b, (x, y,g) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      g = g.to(device, non_blocking=True)
      chrono._done("eval load", time.time() - end)
      with chrono.measure("eval fprop"):
        logits = model(x)
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1 = accuracy(logits, y)
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.append(top1.detach().cpu().item())
    end = time.time()

  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%},")
  logger.flush()
  return np.mean(all_top1)

def train_model(args):
  logger = bit_common.setup_logger(args)
  logger.info(f"Fine-tuning {args.model_type} on {args.dataset}")
  torch.backends.cudnn.benchmark = True
  logger.info(f"Going to train on {args.device}")
  args.train_batch_size = args.train_batch_size // args.batch_split
  train_loader, valid_loader = get_loader_train(args)

  logger.info(f"Loading model from {args.model_type}.npz")
  model = models.KNOWN_MODELS[args.model_type](head_size=2, zero_head=True)
  model.load_from(np.load("bit_pretrained_models/"+args.model_type+".npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)
  step = 0
  optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
  model_checkpoint_dir = pjoin(args.output_dir, args.name, args.dataset, args.model_arch)
  savename = pjoin(model_checkpoint_dir,args.model_type + ".pth.tar")
  if os.path.exists(model_checkpoint_dir) != True:
         os.makedirs(model_checkpoint_dir, exist_ok=True)

  model = model.to(args.device)
  optim.zero_grad()

  model.train()
  cri = torch.nn.CrossEntropyLoss().to(args.device)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  end = time.time()
  best_acc = 0
  with lb.Uninterrupt() as u:
    for x, y,g in recycle(train_loader):
    
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break

      # Schedule sending to GPU(s)
      x = x.to(args.device, non_blocking=True)
      y = y.to(args.device, non_blocking=True)
      g = g.to(args.device, non_blocking=True)
      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_loader.dataset), base_lr = 0.003)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr
      # compute output
      with chrono.measure("fprop"):
        logits = model(x)
        c = cri(logits, y)  
        c_num = float(c.data.cpu().numpy())  

      # Accumulate grads
      with chrono.measure("grads"):
        (c / args.batch_split).backward()
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          optim.zero_grad()
        step += 1
        accum_steps = 0
        if args.eval_every and step % args.eval_every == 0:
          acc = run_eval(model, valid_loader, args.device, chrono, logger, step)
          logger.info("Saved model checkpoint")
          if best_acc < acc:
                        best_acc = acc
                        torch.save({
                              "step": step,
                              "model": model.state_dict(),
                              "optim" : optim.state_dict(),
                          }, savename)

      end = time.time()


  logger.info("Best Accuracy: \t%f" % best_acc)
  logger.info("End Training!")

  logger.info(f"Timings:\n{chrono}")
