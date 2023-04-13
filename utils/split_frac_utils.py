from typing import Dict
from torch.utils.data import Dataset
from datasets.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    Generic_MIL_Dataset,
    save_splits
)
from pathlib import Path

def get_split_dataset(
        label_csv_path: str,
        label_dict: Dict,
        taskname: str = "tumor_vs_normal",
        seed: int = 1,
    ) -> Dataset:
    if taskname == 'tumor_vs_normal':
        dataset = Generic_WSI_Classification_Dataset(
            csv_path = Path(label_csv_path),
            shuffle = False, 
            seed = seed, 
            print_info = True,
            label_dict = label_dict,
            patient_strat=True,
            ignore=[]
        )
    elif taskname == 'tumor_subtyping':
        dataset = Generic_WSI_Classification_Dataset(
            csv_path = Path(label_csv_path),
            shuffle = False, 
            seed = seed, 
            print_info = True,
            label_dict = label_dict,
            patient_strat= True,
            patient_voting='maj',
            ignore=[]
        )
    else:
        raise NotImplementedError

    return dataset