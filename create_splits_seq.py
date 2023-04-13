
import os, sys
from pathlib import Path
import yaml

import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

from utils.split_frac_utils import (
    get_split_dataset
)

parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument(
    '--config',
    type = str,
    default="./conf/conf.yaml",
	help='Creating splits for whole slide classification'
)
args = parser.parse_args()

conf_path = Path(args.config)
with open(conf_path, 'r', encoding="utf-8") as f:
    conf = yaml.safe_load(f)
f.close()

dataset= get_split_dataset(
    label_csv_path=conf["label_csv_path"],
    label_dict=conf["label_dict"],
    taskname=conf["task"],
    seed=conf["seed"],
)

n_classes = conf["n_classes"]
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * conf["val_frac"]).astype(int)
test_num = np.round(num_slides_cls * conf["test_frac"]).astype(int)

if __name__ == '__main__':
    if conf["label_frac"] > 0:
        label_fracs = [conf["label_frac"]]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(conf["task"]) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(
            k=conf['k'],
            val_num=val_num,
            test_num=test_num,
            label_frac=lf
        )

        for i in range(conf['k']):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



