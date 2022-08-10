import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
args = get_args()
print(args)
dataset = get_dataset(
    transform=get_aug(train=True, **args.aug_kwargs),
    train=True,
    **args.dataset_kwargs)

print(len(dataset.data))

