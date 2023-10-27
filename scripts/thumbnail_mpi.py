import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
import math
from multiprocessing import get_context
import os
import time
from feabas.time_region import time_region
from feabas import config, logging
from feabas import mipmap, common, material
from feabas.mipmap import mip_map_one_section
from thumbnail_main import setup_globals, parse_args, downsample_main, setup_pair_names, align_main
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()

if __name__=='__main__':
    args = parse_args()

    root_dir, generate_settings, num_cpus, thumbnail_configs, thumbnail_mip_lvl, mode, num_workers, nthreads, thumbnail_dir, stitch_tform_dir, img_dir, mat_mask_dir, reg_mask_dir, manual_dir, match_dir, feature_match_dir = setup_globals(args)
    if mode=='downsample':
        stitch_conf = config.stitch_configs()['rendering']
        driver = stitch_conf.get('driver', 'image')
        if driver == 'image':        
            min_mip = thumbnail_configs.get('min_mip', 0)
            meta_dir = os.path.join(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt')
            meta_list = sorted(glob.glob(meta_dir, recursive=True))
            assert len(meta_list)>0, f"did not find any metadata.txt files in {os.path.abspath(meta_dir)}"
            sections_per_rank = int(math.ceil(len(meta_list)/NUMRANKS))
            if RANK!=(NUMRANKS-1):
                arg_indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
            else:
                arg_indx = slice(RANK*sections_per_rank, len(meta_list), 1)
            downsample_main(meta_list = meta_list[arg_indx])
        else:
            raise NotImplementedError("mpi not setup for non-image downsampling")  
    elif mode == 'alignment':
        compare_distance = thumbnail_configs.pop('compare_distance', 1)
        imglist, bname_list, pairnames = setup_pair_names(img_dir,root_dir,  compare_distance)
        sectionpairs_per_rank = int(math.ceil(len(pairnames)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            arg_indx = slice(RANK*sectionpairs_per_rank, (RANK+1)*sectionpairs_per_rank, 1)
        else:
            arg_indx = slice(RANK*sectionpairs_per_rank, len(pairnames), 1)            
        pairnames = pairnames[arg_indx]
        align_main(pairnames=pairnames)
        time_region.log_summary()