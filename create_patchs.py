import sys
import os
import cv2
import pandas as pd
from pathlib import Path
import yaml
import argparse

sys.dont_write_bytecode = True

from utils.create_patchs import (
    seg_and_patch
)

parser = argparse.ArgumentParser(description='seg and patch')

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

if __name__ == "__main__":
    # create directories
    patch_save_dir = Path(conf["save_dir"], "patches")
    mask_save_dir = Path(conf["save_dir"], "masks")
    stitch_save_dir = Path(conf["save_dir"], "stitches")

    if conf["process_list"]:
        process_list = Path(conf["save_dir"], conf["process_list"])
    else:
        process_list = None

    print('source: ', conf["source"])
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {
        'source': conf["source"], 
        'save_dir': conf["save_dir"],
        'patch_save_dir': patch_save_dir, 
        'mask_save_dir' : mask_save_dir, 
        'stitch_save_dir': stitch_save_dir
    }

    for key, val in directories.items():
        # print("{} : {}".format(key, val))
        if key in ['source']:
            continue

        os.makedirs(val, exist_ok=True)

    seg_params = conf["seg_params"]
    filter_params = conf["filter_params"]
    vis_params = conf["vis_params"]
    patch_params = conf["patch_params"]
    
    if conf["preset"]:  # default: None
        preset_df = pd.read_csv(Path('presets', conf["preset"]))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]
    
    parameters = {
        'seg_params': seg_params,
        'filter_params': filter_params,
        'patch_params': patch_params,
        'vis_params': vis_params
    }

    df = seg_and_patch(
        **directories,
        **parameters,
        conf=conf,
        use_default_params=False,
        save_mask = True, 
        process_list=process_list,
    )

    if os.path.basename(conf_path) == "conf.yaml" :
        conf_path = Path(args.config)
        with open(conf_path, 'r', encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        f.close()
        conf["process_list"] = "process_list_edited.csv"
        
        save_csvpath = Path(conf["save_dir"], conf["process_list"])
        df.to_csv(save_csvpath)
        
        save_yamlpath = Path(conf["save_dir"], "create_patchs.yaml")
        with open(save_yamlpath, 'w') as f:
            yaml.dump(conf, f, allow_unicode=True, default_flow_style=False)
        f.close()