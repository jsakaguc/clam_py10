from __future__ import print_function

import os, sys
from pathlib import Path
import random
from typing import Dict

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# internal imports
from datasets.dataset_generic import Generic_MIL_Dataset


def seed_torch(seed=7, device="cpu"):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_GenericMILdataset(
        taskname: str,
        label_csv_path: Path,
        data_root_dir: Path,
        label_dict: Dict,
        seed: int = 1,
    ):
    if taskname == 'tumor_vs_normal':
        dataset = Generic_MIL_Dataset(
            csv_path = label_csv_path,
            data_dir= data_root_dir / 'tumor_vs_normal_resnet_features',
            shuffle = False, 
            seed = seed, 
            print_info = True,
            label_dict = label_dict,
            patient_strat=False,
            ignore=[]
        )

    elif taskname == 'tumor_subtyping':
        dataset = Generic_MIL_Dataset(
            csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
            data_dir= data_root_dir / 'tumor_subtyping_resnet_features',
            shuffle = False, 
            seed = seed, 
            print_info = True,
            label_dict = label_dict,
            patient_strat= False,
            ignore=[]
        )   
    else:
        raise NotImplementedError
    
    return dataset