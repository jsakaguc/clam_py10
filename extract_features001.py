import torch
import torch.nn as nn
from math import floor
import os
import sys
import yaml
from pathlib import Path
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import (
	Dataset_All_Bags,
	Whole_Slide_Bag_FP
)
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
from torch.nn import Module

sys.path.append("utils/openslide/openslide")
import openslide
from openslide import OpenSlide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device:", device)

def compute_w_loader(
    file_path: Path,
    output_path: Path,
    wsi: OpenSlide,
    model: Module,
 	batch_size: int = 8,
    verbose: int = 0,
    print_every: int = 20,
    pretrained: bool = True, 
	custom_downsample: int = 1,
    target_patch_size: int = -1
) -> Path:
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(
		file_path=file_path,
		wsi=wsi,
		pretrained=pretrained, 
		custom_downsample=custom_downsample,
		target_patch_size=target_patch_size
	)

	x, y = dataset[0]
	
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		**kwargs,
		collate_fn=collate_features
	)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


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


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = Path(conf["csv_path"])
	if csv_path.exists() is False:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(conf["feat_dir"], exist_ok=True)
	os.makedirs(os.path.join(conf["feat_dir"], 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(conf["feat_dir"], 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(conf["feat_dir"], 'pt_files'))

	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(conf["slide_ext"])[0]
		data_class = bags_dataset.get_classes(bag_candidate_idx)
		
		bag_name = slide_id+'.h5'
		h5_file_path = Path(conf["data_h5_dir"], 'patches', bag_name)
		
		slide_file_path = Path(conf["data_slide_dir"], data_class, slide_id+conf["slide_ext"])
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not conf["no_auto_skip"] and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = Path(conf["feat_dir"], 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
 	
		output_file_path = compute_w_loader(
			file_path=h5_file_path,
			output_path=output_path,
			wsi=wsi,
			model=model,
			batch_size=conf["batch_size"],
			verbose=1,
			print_every=20, 
			custom_downsample=conf["custom_downsample"],
			target_patch_size=conf["target_patch_size"]
		)
			
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(conf["feat_dir"], 'pt_files', bag_base+'.pt'))