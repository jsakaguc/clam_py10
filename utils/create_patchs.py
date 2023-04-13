import os, sys
from typing import Dict, Union, List, Tuple
from pathlib import Path
import numpy as np
import time
import argparse
import pdb
import pandas as pd

from wsi_core.batch_process_utils import initialize_df
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords

def stitching(file_path, wsi_object, downscale = 64):
    start = time.time()

    heatmap = StitchCoords(
        hdf5_file_path=file_path,
        wsi_object=wsi_object,
        downscale=downscale,
        bg_color=(0,0,0),
        alpha=-1,
        draw_grid=False
    )
    total_time = time.time() - start

    return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
    source: Path,
    save_dir: Path,
    patch_save_dir: Path,
    mask_save_dir: Path,
    stitch_save_dir: Path,
    seg_params: Dict = {
        'seg_level': -1,
        'sthresh': 8,
        'mthresh': 7,
        'close': 4,
        'use_otsu': False,
        'keep_ids': 'none',
        'exclude_ids': 'none'
    },
    conf: Dict = {},
    filter_params: Dict = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
    vis_params: Dict = {'vis_level': -1, 'line_thickness': 500},
    patch_params: Dict = {'use_padding': True, 'contour_fn': 'four_pt'},
    use_default_params: bool = False,
    save_mask: bool = True,
    process_list: Union[None, Path] = None
) -> Tuple[float, float]:
    source = Path(source)
    patch_size: int = conf["patch_size"]
    seg: bool = conf["seg"]
    step_size: int = conf["step_size"]
    stitch: bool = conf["stitch"]
    patch: bool = conf["patch"]
    patch_level: int = conf["patch_level"]
    auto_skip: bool = conf["no_auto_skip"]
    seg_width: int = conf["seg_width"]

    slides = []
    dirnames = []
    for ext in conf["image_ext_reg"]:
        for f in source.glob(f"**/*{ext}"):
            slides.append(Path(f).name)
            dirnames.append(Path(f).parent.name)
    print("slides length:", len(slides))

    if process_list is None:
        df = initialize_df(
            slides,
            seg_params,
            filter_params, 
            vis_params, 
            patch_params
        )
    else:
        print("read process_list:", process_list)
        df = pd.read_csv(process_list)
        df = initialize_df(
            df,
            seg_params,
            filter_params,
            vis_params,
            patch_params
        )
    df["classes"] = dirnames
    # print(df.head())

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(
             **{
                'a_t': np.full(
                    (len(df)), int(filter_params['a_t']), dtype=np.uint32
                ),
                'a_h': np.full(
                    (len(df)), int(filter_params['a_h']), dtype=np.uint32
                ),
                'max_n_holes': np.full(
                    (len(df)), int(filter_params['max_n_holes']), dtype=np.uint32
                ),
                'line_thickness': np.full(
                    (len(df)), int(vis_params['line_thickness']), dtype=np.uint32
                ),
                'contour_fn': np.full(
                    (len(df)), patch_params['contour_fn']
                )
            }
        )

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.
    
    process_csv_path = Path(save_dir, "process_list_autogen.csv")
    for i in range(total):
        df.to_csv(process_csv_path, index=False)
        idx = process_stack.index[i]
        slide = Path(process_stack.loc[idx, 'slide_id'])
        print("\n\nprogress: {:.2f}, {}/{}".format((i+1)/total, i+1, total))
        print(f"processing {slide}")

        df.loc[idx, 'process'] = 0
        slide_id= slide.stem
        slide_h5_path = Path(patch_save_dir, f"{slide_id}.h5")
        if auto_skip and slide_h5_path.is_file():
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = next(source.glob(f"**/{slide}"))
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}


            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                if process_list is not None:
                    current_vis_params['vis_level'] = best_level
                else:
                    current_vis_params['vis_level'] = current_seg_params['seg_level']

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level
        
        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue
        
        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(
                WSI_object=WSI_object,
                seg_params=current_seg_params,
                filter_params=current_filter_params
            )

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update({
                'patch_level': patch_level,
                'patch_size': patch_size,
                'step_size': step_size,
                'save_path': patch_save_dir
            })
            file_path, patch_time_elapsed = patching(
                WSI_object=WSI_object,
                **current_patch_params,
            )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(
                    file_path,
                    WSI_object,
                    downscale=64
                )
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed + 1e-7
        patch_times += patch_time_elapsed + 1e-7
        stitch_times += stitch_time_elapsed + 1e-7

    total += 1e-7
    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(process_csv_path, index=False)
    print("average segmentation time in s per slide: {:.1f}s".format(seg_times))
    print("average patching time in s per slide: {:.1f}s".format(patch_times))
    print("average stiching time in s per slide: {:.1f}s".format(stitch_times))

    return df
