# Are Vision Transformers Robust to Spurious Correlations?
This codebase provides a Pytorch implementation for the paper: Are Vision Transformers Robust to Spurious Correlations? . 

## Abstract
Deep neural networks may be susceptible to learning spurious correlations that hold on average but not in atypical test samples. As with the recent emergence of vision transformer (ViT) models, it remains underexplored how spurious correlations are manifested in such architectures. In this paper, we systematically investigate the robustness of vision transformers to spurious correlations on three challenging benchmark datasets and compare their performance with popular CNNs. Our study reveals that when pre-trained on a sufficiently large dataset, ViT models are more robust to spurious correlations than CNNs. Key to their success is the ability to generalize better from the examples where spurious correlations do not hold. Further, we perform extensive ablations and experiments to understand the role of the self-attention mechanism in providing robustness under spuriously correlated environments. We hope that our work will inspire future research on further understanding the robustness of ViT models.

## Required Packages
Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.9 and Pytorch 1.6. Besides, the following packages are required to be installed:
* Scipy
* Numpy
* Sklearn
* Pandas
* tqdm
* pillow
* timm

## Pre-trained Checkpoints
In our experiments, for ViT models we use the pre-trained checkpoints provided with the [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm) library. Pre-trained checkpoints for BiT models can be downloaded following the instructions in the [official repo](https://github.com/google-research/big_transfer/blob/master/README.md). Please download place the checkpoints for BiTs in `bit_pretrained_models` folder.

## Datasets

In our study, we use the following challenging benchmarks :
  - WaterBirds:  Similar to the construction in [Group_DRO](https://github.com/kohpangwei/group_DRO), this dataset is constructed by cropping out birds from photos in the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) and transferring them onto backgrounds from the Places dataset (Zhou et al., 2017).
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Large-scale CelebFaces Attributes Dataset. The data we used for this task is listed in `datasets/celebA/celebA_split.csv`, and after downloading the dataset, please place the images in the folder of `datasets/celebA/img_align_celeba/`. 
  - ColorMINST:  A colour-biased version of the original [MNIST](http://yann.lecun.com/exdb/mnist/) Dataset. 

## Quick Start
To run the experiments, you need to first download and place the pretrained model checkpoints and datasets in the specificed folders as instructed in [Pre-trained Checkpoints](#Pre-trained Checkpoints) and [Datasets](#Datasets). We provide the following commands and general descriptions for related files.

### WaterBirds
* `datasets/waterbirds_dataset.py`: provides the dataloader for Waterbirds dataset.
The code expects the following files/folders in the `[root_dir]/datasets` directory:

- `waterbird_complete95_forest2water2/`

You can download a tarball of this dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). The Waterbirds dataset can also be accessed through the [WILDS package](https://github.com/p-lambda/wilds), which will automatically download the dataset.

To train ViT model (ViT-S_16) on Waterbirds dataset, run the following command:
```
python train.py --name waterbirds_exp --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --warmup_steps 200 --num_steps 500 --learning_rate 0.03 --batch_split 1 --img_size 384
```
Similarly, sample command to run BiT model on Watervirds dataset:
```
python train.py --name waterbirds_exp --model_arch BiT --model_type BiT-M-R50x1 --dataset waterbirds --learning_rate 0.003--batch_split 1 --img_size 384
```
Notes for some of the arguments:
* `--name`: Name to identify the checkpoints. Users are welcome to use other names for convenience.
* `--model_arch` : Model architecture to be used for training. Users need to specify `ViT` for Vision Transformers or `BiT` for Big-Transfer models.
* `--model_type` : Model variant to be used for training. Please check the table below.
* `--warmup_steps` : Specifies the number of warmup steps used for training ViT models. For ViT-S_16 and ViT-Ti_16, this is set as 200 whereas for ViT-B_16 set this as 500.
* `--num_steps` : Specifies the total number of global steps used for training ViT models. For ViT-S_16 and ViT-Ti_16, this is set as 500 whereas for ViT-B_16 set this as 2000.
* `--batch_split`: The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `batch_split`.

| Model        | model_arch | model_type   | #params |
|--------------|------------|--------------|---------|
| ViT-B/16     | ViT        | ViT-B_16     | 86.1 M  |
| ViT-S/16     | ViT        | ViT-S_16     | 21.8 M  |
| ViT-Ti/16    | ViT        | ViT-Ti_16    | 5.6 M   |
| BiT-M-R50x3  | BiT        | BiT-M-R50x3  | 211 M   |
| BiT-M-R101x1 | BiT        | BiT-M-R101x1 | 42.5 M  |
| BiT-M-R50x1  | BiT        | BiT-M-R50x1  | 23.5 M  |

To generate accuracy metrics for ViT model(ViT-S_16) on train and test data (worst-group accuracy), run the following command :
```
python evaluate.py --name waterbirds_exp --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --batch_size 64 --img_size 384
```

#### Spurious OOD evaluation
To generate the OOD dataset, users need to run `datasets/generate_placebg.py` which subsamples background images of specific types as the OOD data. You can simply run `python generate_placebg.py` to generate the OOD dataset, and it will be stored as `datasets/ood_datasets/placesbg/`. Note: Before the generation of OOD dataset, users need to download and change the path of CUB dataset and Places dataset.
To obtain spurious OOD evaluation for for ViT model(ViT-S_16), run the following command:
```
python ood_eval.py --name waterbirds_exp --model_arch ViT --model_type ViT-S_16 --id_dataset waterbirds --batch_size 64 --img_size 384
```

### ColorMNIST
* `datasets/color_mnist.py` downloads the original MNIST and applies colour biases on images by itself. No extra preparation is needed on the user side.

Here is an example to train ViT model (ViT-S_16) on the ColorMNIST Dataset:

```
python train.py --name cmnist_exp --model_arch ViT --model_type ViT-S_16 --dataset cmnist --warmup_steps 200 --num_steps 500 --learning_rate 0.03 --batch_split 1 --img_size 224
```
To generate accuracy metrics for ViT model(ViT-S_16) on train and test data (worst-group accuracy), run the following command :
```
python evaluate.py --name cmnist_exp --model_arch ViT --model_type ViT-S_16 --dataset cmnist --batch_size 64 --img_size 224
```
#### Spurious OOD evaluation
Spurious OOD data for CMNIST can be downloaded [here](https://www.dropbox.com/s/kqqm9doda33f4tt/partial_color_mnist_0%261.zip?dl=0) and place it under `datasets/ood_data`.

To obtain spurious OOD evaluation for for ViT model(ViT-S_16), run the following command:
```
python ood_eval.py --name cmnist_exp --model_arch ViT --model_type ViT-S_16 --id_dataset cmnist --batch_size 64 --img_size 224
```

### CelebA
* `datasets/celebA_dataset.py`: provides the dataloader for CelebA datasets and OOD datasets.

Here is an example to train ViT model (ViT-S_16) on the ColorMNIST Dataset:

```
python train.py --name celeba_exp --model_arch ViT --model_type ViT-S_16 --dataset celebA --warmup_steps 200 --num_steps 500 --learning_rate 0.03 --batch_split 1 --img_size 384
```
To generate accuracy metrics for ViT model(ViT-S_16) on train and test data (worst-group accuracy), run the following command :
```
python evaluate.py --name celeba_exp --model_arch ViT --model_type ViT-S_16 --dataset celebA --batch_size 64 --img_size 384
```
#### Spurious OOD evaluation
The meta data for this dataset has already been included in the provided CelebA zip file as `datasets/CelebA/celebA_ood.csv`.
To obtain spurious OOD evaluation for for ViT model(ViT-S_16), run the following command:
```
python ood_eval.py --name celeba_exp --model_arch ViT --model_type ViT-S_16 --id_dataset celebA --batch_size 64 --img_size 384
```
## References
Some parts of the codebase are adapted from [GDRO](https://github.com/kohpangwei/group_DRO), [Spurious_OOD](https://github.com/deeplearning-wisc/Spurious_OOD), [big_transfer](https://github.com/google-research/big_transfer) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch).

