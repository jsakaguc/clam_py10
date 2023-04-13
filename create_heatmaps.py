from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
from typing import Dict
from pathlib import Path
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

parser = argparse.ArgumentParser(description='Heatmap inference script')

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

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key]
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(conf: Dict, config_dict: Dict) -> Dict:
	if conf["save_exp_code"] is not None:
		config_dict['exp_arguments']['save_exp_code'] = conf["save_exp_code"]

	if conf["overlap"] is not None:
		config_dict['patching_arguments']['overlap'] = conf["overlap"]

	return config_dict

def prepare_process_list(conf: Dict) -> pd.DataFrame:
	save_csvpath = Path(conf["save_dir"], conf["process_list"])
	print("process_list: ", save_csvpath)
	df = pd.read_csv(save_csvpath)
	df["label"] = df["classes"]
	df = df[["slide_id", "process", "label"]]
	return df

if __name__ == '__main__':
	config_path = Path(conf["heatmap_config"])
	conf_heatmap = yaml.safe_load(
		open(config_path, 'r', encoding="utf-8")
	)
	conf_heatmap = parse_config_dict(conf, conf_heatmap)

	for key, value in conf_heatmap.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))

	# decision = input('Continue? Y/N ')
	decision = 'y'
	if decision in ['Y', 'y', 'Yes', 'yes']:
		pass
	elif decision in ['N', 'n', 'No', 'NO']:
		exit()
	else:
		raise NotImplementedError

	args = conf_heatmap
	# patching_arguments
	patch_args = {
		"patch_size": conf["patch_size"],
		"overlap": conf["overlap"],
		"patch_level": conf["patch_level"],
		"custom_downsample": conf["custom_downsample"]
	}
	# data_arguments
	data_args = {
		"data_dir": conf["data_slide_dir"],
		"data_dir_key": conf["data_arguments"]["data_dir_key"],
		"process_list": Path(conf["save_dir"], conf["process_list"]),
		"preset": conf["preset"],
		"slide_ext": conf["slide_ext"],
		"label_dict": conf["label_dict"]
	}
	# model_arguments
	model_args = {
		"ckpt_path": conf["model_arguments"]["ckpt_path"],
		"initiate_fn": conf["model_arguments"]["initiate_fn"],
		"drop_out": conf["model_arguments"]["drop_out"],
		"model_type": conf["model_type"],
		"model_size": conf["model_size"],
		"n_classes": conf["n_classes"] 
	}
	# exp_arguments
	exp_args = {
		"n_classes": conf["n_classes"],
		"save_exp_code": conf["exp_arguments"]["save_exp_code"],
		"raw_save_dir": conf["exp_arguments"]["raw_save_dir"],
		"production_save_dir": conf["exp_arguments"]["production_save_dir"],
		"batch_size": conf["batch_size"]
	}
	# heatmap_arguments
	heatmap_args = conf["heatmap_arguments"]

	# sample_arguments
	sample_args = conf['sample_arguments']

	patch_size = tuple([patch_args["patch_size"] for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args["overlap"])).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
		patch_size[0], patch_size[1],
		patch_args["overlap"],
		step_size[0], step_size[1])
	)


	preset = Path("./presets", data_args["preset"])
	def_seg_params = {
		'seg_level': -1,
		'sthresh': 15,
		'mthresh': 11,
		'close': 2,
		'use_otsu': False,
		'keep_ids': 'none',
		'exclude_ids':'none'
	}

	def_filter_params = {
		'a_t':50.0,
		'a_h': 8.0,
		'max_n_holes':10
	}

	def_vis_params = {
		'vis_level': -1,
		'line_thickness': 250
	}

	def_patch_params = {
		'use_padding': True,
		'contour_fn': 'four_pt'
	}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if data_args["process_list"] is None:
		if isinstance(data_args["data_dir"], list):
			slides = []
			for data_dir in data_args["data_dir"]:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args["data_dir"]))
		slides = [slide for slide in slides if data_args["slide_ext"] in slide]
		df = initialize_df(
			slides,
			def_seg_params,
			def_filter_params,
			def_vis_params,
			def_patch_params,
			use_heatmap_args=False
		)
	else:
		# df = pd.read_csv(Path('heatmaps/process_lists', data_args["process_list"]))
		heatmap_dataset = prepare_process_list(conf)
		df = initialize_df(
			heatmap_dataset,
			def_seg_params,
			def_filter_params,
			def_vis_params,
			def_patch_params,
			use_heatmap_args=False
		)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args["ckpt_path"]
	print(f"ckpt path: {ckpt_path}")

	if model_args["initiate_fn"] == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError


	feature_extractor = resnet50_baseline(pretrained=True)
	feature_extractor.eval()
	print('Done!')

	label_dict =  data_args["label_dict"]
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	os.makedirs(exp_args["production_save_dir"], exist_ok=True)
	os.makedirs(exp_args["raw_save_dir"], exist_ok=True)
	blocky_wsi_kwargs = {
		'top_left': None,
		'bot_right': None,
		'patch_size': patch_size,
		'step_size': patch_size,
		'custom_downsample':patch_args["custom_downsample"],
		'level': patch_args["patch_level"],
		'use_center_shift': heatmap_args["use_center_shift"]
	}

	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		if data_args["slide_ext"] not in slide_name:
			slide_name += data_args["slide_ext"]
		print('\nprocessing: ', slide_name)

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.replace(data_args["slide_ext"], '')

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = Path(
			exp_args["production_save_dir"],
			exp_args["save_exp_code"],
			str(grouping)
		)
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = Path(
			exp_args["raw_save_dir"],
			exp_args["save_exp_code"],
			str(grouping),
			slide_id
		)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		if heatmap_args["use_roi"]:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None

		print(f"slide id: {slide_id}")
		print(f"top left: {top_left},  bot right: {bot_right}")

		if isinstance(data_args["data_dir"], str):
			slide_path = Path(
				data_args["data_dir"],
				process_stack.loc[i, 'label'],
				slide_name
			)
		elif isinstance(data_args["data_dir"], dict):
			data_dir_key = process_stack.loc[i, data_args["data_dir_key"]]
			slide_path = Path(
				data_args["data_dir"][data_dir_key],
				process_stack.loc[i, 'label'],
				slide_name
			)
		else:
			raise NotImplementedError

		mask_file = Path(r_slide_save_dir, slide_id+'_mask.pkl')

		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))

		print('Initializing WSI object')
		wsi_object = initialize_wsi(
			slide_path,
			seg_mask_path=mask_file,
			seg_params=seg_params,
			filter_params=filter_params
		)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[patch_args["patch_level"]]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((
			np.array(patch_size) * \
			np.array(wsi_ref_downsample) * \
			patch_args["custom_downsample"]).astype(int)
		)

		block_map_save_path = Path(r_slide_save_dir, f"{slide_id}_blockmap.h5")
		mask_path = Path(r_slide_save_dir, f"{slide_id}_mask.jpg")
	
		if vis_params['vis_level'] < 0:
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			vis_params['vis_level'] = best_level
	
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)

		features_path = Path(r_slide_save_dir, f"{slide_id}.pt")
		h5_path = Path(r_slide_save_dir, f"{slide_id}.h5")


		##### check if h5_features_file exists ######
		if h5_path.exists() is False :
			_, _, wsi_object = compute_from_patches(
				wsi_object=wsi_object,
				model=model,
				feature_extractor=feature_extractor,
				batch_size=exp_args["batch_size"],
				**blocky_wsi_kwargs,
				attn_save_path=None,
				feat_save_path=h5_path,
				ref_scores=None
			)

		##### check if pt_features_file exists ######
		if features_path.exists() is False:
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()

		# load features
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)

		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(
			model,
			features,
			label,
			reverse_label_dict,
			exp_args["n_classes"]
		)
		del features

		if block_map_save_path.exists() is False:
			file = h5py.File(h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

		# save top 3 predictions
		for c in range(exp_args["n_classes"]):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		os.makedirs('heatmaps/results/', exist_ok=True)
		if data_args["process_list"] is not None:
			process_stack.to_csv(
				f"heatmaps/results/{data_args['process_list'].stem}.csv",
				index=False
			)
		else:
			process_stack.to_csv(
				'heatmaps/results/{}.csv'.format(exp_args["save_exp_code"]),
				index=False
			)

		h5_file = h5py.File(block_map_save_path, 'r')
		dset = h5_file['attention_scores']
		coord_dset = h5_file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		h5_file.close()

		for sample in sample_args["samples"]:
			if sample['sample']:
				tag = f"label_{label}_pred_{Y_hats[0]}"
				sample_save_dir =  Path(
					exp_args["production_save_dir"],
					exp_args["save_exp_code"],
				)
				sample_save_dir = sample_save_dir / 'sampled_patches' / str(tag) / sample['name']
				
				os.makedirs(sample_save_dir, exist_ok=True)
				print(f"sampling {sample['name']}")

				sample_results = sample_rois(
					scores,
					coords,
					k=sample['k'],
					mode=sample['mode'],
					seed=sample['seed'],
					score_start=sample.get('score_start', 0),
					score_end=sample.get('score_end', 1)
				)

				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(
						tuple(s_coord),
						patch_args["patch_level"],
						(patch_args["patch_size"], patch_args["patch_size"])
					).convert('RGB')

					filename = f"{idx}_{slide_id}_x_{s_coord[0]}_y_{s_coord[1]}_a_{s_score:.3f}.png"
					patch.save(os.path.join(sample_save_dir, filename))

		wsi_kwargs = {
			'top_left': top_left,
			'bot_right': bot_right,
			'patch_size': patch_size,
			'step_size': step_size,
			'custom_downsample':patch_args["custom_downsample"],
			'level': patch_args["patch_level"],
			'use_center_shift': heatmap_args["use_center_shift"]
		}

		heatmap_save_name = f"{slide_id}_blockmap.tiff"
		if Path(r_slide_save_dir, heatmap_save_name).exists():
			pass
		else:
			heatmap = drawHeatmap(
				scores,
				coords,
				slide_path,
				wsi_object=wsi_object,
				cmap=heatmap_args["cmap"],
				alpha=heatmap_args["alpha"],
				use_holes=True,
				binarize=False,
				vis_level=-1,
				blank_canvas=False,
				thresh=-1,
				patch_size=vis_patch_size,
				convert_to_percentiles=True
			)

			heatmap.save(os.path.join(r_slide_save_dir, f"{slide_id}_blockmap.png"))
			del heatmap

		filename = f"{slide_id}_{patch_args['overlap']}_roi_{heatmap_args['use_roi']}.h5"
		save_path = Path(r_slide_save_dir, filename)

		if heatmap_args["use_ref_scores"]:
			ref_scores = scores
		else:
			ref_scores = None
		if heatmap_args["calc_heatmap"]:
			compute_from_patches(
				wsi_object=wsi_object,
				clam_pred=Y_hats[0],
				model=model,
				feature_extractor=feature_extractor,
				batch_size=exp_args["batch_size"],
				**wsi_kwargs,
				attn_save_path=save_path,
				ref_scores=ref_scores[:, 0]
			)

		if save_path.exists() is False:
			print(f"heatmap {save_path} not found")
			if heatmap_args["use_roi"]:
				save_path_full = Path(
					r_slide_save_dir,
					f"{slide_id}_{patch_args['overlap']}_roi_False.h5")
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		h5_file = h5py.File(save_path, 'r')
		dset = h5_file['attention_scores']
		coord_dset = h5_file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		h5_file.close()

		heatmap_vis_args = {
			'convert_to_percentiles': True,
			'vis_level': heatmap_args["vis_level"],
			'blur': heatmap_args["blur"],
			'custom_downsample': heatmap_args["custom_downsample"]
		}
		if heatmap_args["use_ref_scores"]:
			heatmap_vis_args['convert_to_percentiles'] = False


		f_1 = f"{slide_id}_{float(patch_args['overlap'])}_roi_{int(heatmap_args['use_roi'])}"
		f_2 = f"blur_{int(heatmap_args['blur'])}_rs_{int(heatmap_args['use_ref_scores'])}"
		f_3 = f"bc_{int(heatmap_args['blank_canvas'])}_a_{float(heatmap_args['alpha'])}"
		f_4 = f"l_{int(heatmap_args['vis_level'])}_bi_{int(heatmap_args['binarize'])}"
		f_5 = f"{float(heatmap_args['binary_thresh'])}.{heatmap_args['save_ext']}"
		heatmap_save_name = f"{f_1}_{f_2}_{f_3}_{f_4}_{f_5}"
		if Path(p_slide_save_dir, heatmap_save_name).exists():
			pass
		else:
			heatmap = drawHeatmap(
				scores,
				coords,
				slide_path,
				wsi_object=wsi_object,
				cmap=heatmap_args["cmap"],
				alpha=heatmap_args["alpha"],
				**heatmap_vis_args,
				binarize=heatmap_args["binarize"],
				blank_canvas=heatmap_args["blank_canvas"],
				thresh=heatmap_args["binary_thresh"],
				patch_size=vis_patch_size,
				overlap=patch_args["overlap"],
				top_left=top_left,
				bot_right=bot_right
			)
			heatmap_savepath = Path(p_slide_save_dir, heatmap_save_name)
			if heatmap_args["save_ext"]== 'jpg':
				heatmap.save(heatmap_savepath , quality=100)
			else:
				heatmap.save(heatmap_savepath)

		if heatmap_args["save_orig"]:
			if heatmap_args["vis_level"] >= 0:
				vis_level = heatmap_args["vis_level"]
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = f"{slide_id}_orig_{int(vis_level)}.{heatmap_args['save_ext']}"
			if Path(p_slide_save_dir, heatmap_save_name).exists():
				pass
			else:
				heatmap = wsi_object.visWSI(
					vis_level=vis_level,
					view_slide_only=True,
					custom_downsample=heatmap_args["custom_downsample"]
				)
				heatmap_savepath = Path(p_slide_save_dir, heatmap_save_name)
				if heatmap_args["save_ext"] == 'jpg':
					heatmap.save(heatmap_savepath, quality=100)
				else:
					heatmap.save(heatmap_savepath)
		
	result_yamlpath = Path(exp_args["raw_save_dir"], exp_args["save_exp_code"], 'result_config.yaml')
	with open(result_yamlpath, 'w') as outfile:
		yaml.dump(conf, outfile, default_flow_style=False)


