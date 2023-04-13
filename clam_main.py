from __future__ import print_function

import argparse
import os, sys
from pathlib import Path
from typing import Dict
import yaml
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

from utils.main_utils import (
    seed_torch,
    get_GenericMILdataset
)
from utils.split_frac_utils import get_split_dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(
        conf: Dict,
        results_dir: Path,
        split_dir: Path,
        dataset
    ):
    # create results directory if necessary
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    start = 0 if conf["k_start"] else conf["k_start"]
    end = conf['k'] if conf['k'] else conf['k']

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(seed=conf["seed"], device=device)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path=split_dir / f"splits_{i}.csv"
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(
            datasets=datasets,
            cur=i,
            conf=conf,
            results_dir=results_dir
        )
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = results_dir / 'split_{i}_results.pkl'
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != conf['k']:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(results_dir / save_name)

parser = argparse.ArgumentParser(description='Configurations for WSI Training')

parser.add_argument(
    '--config',
    type = str,
    default="./conf/conf.yaml",
	help='path to config file'
)
args = parser.parse_args()

conf_path = Path(args.config)
with open(conf_path, 'r', encoding="utf-8") as f:
    conf = yaml.safe_load(f)
f.close()

# parser.add_argument('--data_root_dir', type=str, default=None, 
                    # help='data directory')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch(seed=conf["seed"], device=device)

encoding_size = 1024
settings = {
    'num_splits': conf['k'], 
    'k_start': conf["k_start"],
    'k_end': conf["k_end"],
    'task': conf["task"],
    'max_epochs': conf["max_epochs"], 
    'results_dir': conf["results_dir"], 
    'lr': conf["lr"],
    'experiment': conf["exp_code"],
    'reg': conf["reg"],
    'label_frac': conf["label_frac"],
    'bag_loss': conf["bag_loss"],
    'seed': conf["seed"],
    'model_type': conf["model_type"],
    'model_size': conf["model_size"],
    "use_drop_out": conf["drop_out"],
    'weighted_sample': conf["weighted_sample"],
    'opt': conf["opt"]
}

if conf["model_type"] in ['clam_sb', 'clam_mb']:
   settings.update({
       'bag_weight': conf["bag_weight"],
        'inst_loss': conf["inst_loss"],
        'B': conf['B']
    })

print('\nLoad Dataset')


dataset= get_GenericMILdataset(
    label_csv_path=Path(conf["label_csv_path"]),
    label_dict=conf["label_dict"],
    taskname=conf["task"],
    seed=conf["seed"],
    data_root_dir=Path(conf["data_root_dir"]),
)
n_classes = conf["n_classes"]

if n_classes > 2 and conf["model_type"] in ['clam_sb', 'clam_mb']:
    assert conf["subtyping"]

results_dir = Path(conf["results_dir"])
if not results_dir.exists():
    os.mkdir(results_dir)

results_dir = results_dir / f"{conf['exp_code']}_s{conf['seed']}"
if not results_dir.exists():
    os.mkdir(results_dir)

split_dir: str = conf["split_dir"]
if split_dir == '':
    # split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    split_dir = Path("splits", f"{conf['task']}_{int(conf['label_frac'])*100}")
else:
    split_dir = Path("./splits", conf["split_dir"])

print('split_dir: ', split_dir)
assert os.path.isdir(split_dir)

settings.update({'split_dir': conf["split_dir"]})


with open(results_dir / f"experiment_{conf['exp_code']}.txt", 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(
        conf=conf,
        results_dir=results_dir,
        split_dir=split_dir,
        dataset=dataset,
    )
    print("finished!")
    print("end script")


