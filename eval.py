from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import sys
import yaml
from pathlib import Path
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

from utils.main_utils import (
    get_GenericMILdataset
)

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')

parser.add_argument(
    '--config',
    type = str,
    default="./conf/conf.yaml",
	help='path to config file'
)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf_path = Path(args.config)
with open(conf_path, 'r', encoding="utf-8") as f:
    conf = yaml.safe_load(f)
f.close()

conf["save_dir"] = Path("./eval_results", f"EVAL_{conf['save_exp_code']}")
conf["models_dir"] = Path(conf["results_dir"], conf["models_exp_code"])


os.makedirs(conf["save_dir"], exist_ok=True)


if conf["splits_dir"] is None:
    conf["splits_dir"] = Path(conf["models_dir"])

assert os.path.isdir(conf["models_dir"])
assert os.path.isdir(conf["splits_dir"])

settings = {
    'task': conf["task"],
    'split': conf["split"],
    'save_dir': conf["save_dir"], 
    'models_dir': conf["models_dir"],
    'model_type': conf["model_type"],
    'drop_out': conf["drop_out"],
    'model_size': conf["model_size"]
}

with open(conf["save_dir"] / f"eval_experiment_{conf['save_exp_code']}.txt", 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
dataset= get_GenericMILdataset(
    label_csv_path=Path(conf["label_csv_path"]),
    label_dict=conf["label_dict"],
    taskname=conf["task"],
    seed=conf["seed"],
    data_root_dir=Path(conf["data_root_dir"]),
)
n_classes = conf["n_classes"]

start = 0 if conf["k_start"] == -1 else conf["k_start"]
end = conf['k'] if conf["k_end"] == -1 else conf["k_end"]

folds = range(start, end) if conf["fold"] == -1 else range(conf["fold"], conf["fold"]+1)

ckpt_paths = [Path(conf["models_dir"], f"s_{fold}_checkpoint.pt") for fold in folds]
datasets_id = conf["datasets_id"]

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[conf["split"]] < 0:
            split_dataset = dataset
        else:
            datasets = dataset.return_splits(
                from_id=False,
                csv_path=f"{conf['splits_dir']}/splits_{folds[ckpt_idx]}.csv"
            )
            split_dataset = datasets[datasets_id[conf["split"]]]
        model, patient_results, test_error, auc, df  = eval(
            split_dataset,
            conf,
            ckpt_paths[ckpt_idx]
        )
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        save_path = conf["save_dir"] /f"fold_{folds[ckpt_idx]}.csv"
        print(f"save path: {save_path}\n\n")
        df.to_csv(save_path, index=False)

    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_auc,
        'test_acc': all_acc
    })

    if len(folds) != conf['k']:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    
    final_df.to_csv(conf["save_dir"] / save_name)
