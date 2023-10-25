import argparse
import glob
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
import math
from functools import partial
import os
import time
import tensorstore as ts

import feabas
from feabas import config, logging, dal
from stitch_main import parse_args, match_main, setup_globals, render_main, optmization_main
from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()
from feabas.stitcher import Stitcher, MontageRenderer
import numpy as np

if __name__=='__main__':
    args = parse_args()
    root_dir, generate_settings, stitch_configs, num_cpus, mode, num_workers, nthreads, stitch_dir, coord_dir, mesh_dir, match_dir, render_meta_dir=setup_globals(args)

    if mode in ['matching', 'match']:
        coord_regex = os.path.abspath(os.path.join(coord_dir, '*.txt'))
        coord_list = sorted(glob.glob(coord_regex))
        assert len(coord_list) > 0, f"didn't find any txt coord files {coord_regex}"
        sections_per_rank = int(math.ceil(len(coord_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(coord_list), 1)
        print(RANK,"looking at indx", indx)
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_list, match_dir, **stitch_configs)
    elif mode in ['rendering', 'render']:
        stitch_configs_render = stitch_configs['rendering']
        stitch_configs_render.pop('out_dir', '')
        image_outdir = config.stitch_render_dir(root_dir)
        driver = stitch_configs_render.get('driver', 'image')
        if driver == 'image':
            image_outdir = os.path.join(image_outdir, 'mip0')
        tform_regex = os.path.abspath(os.path.join(mesh_dir, '*.h5'))
        tform_list = sorted(glob.glob(tform_regex))
        assert len(tform_list)>0, f"tform list empty, didn't find any h5 files in {tform_regex}"
        sections_per_rank = int(math.ceil(len(tform_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(tform_list), 1)
        print(RANK,"looking at indx", indx)
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        stitch_configs_render.setdefault('meta_dir', render_meta_dir)
        print(f"image_outdir {image_outdir}")
        render_main(tform_list, image_outdir, **stitch_configs_render)        
    elif mode in ['optimization', 'optimize']:
        stitch_configs_opt = stitch_configs['optimization']
        match_regex = os.path.abspath(os.path.join(match_dir, '*.h5'))
        match_list = sorted(glob.glob(match_regex))
        assert len(match_list) > 0, f"match list couldn't find any h5 files {match_regex}"
        sections_per_rank = int(math.ceil(len(match_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(match_list), 1)
        print(RANK,"looking at indx", indx)
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        os.makedirs(mesh_dir, exist_ok=True)
        optmization_main(match_list, mesh_dir, **stitch_configs_opt)