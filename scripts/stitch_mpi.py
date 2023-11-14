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

from feabas.time_region import time_region
import feabas
from feabas import config, logging, dal
from stitch_main import parse_args, match_main, setup_globals, render_main, optmization_main
from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()
from feabas.stitcher import Stitcher, MontageRenderer
import numpy as np

def matching_check_status(coord_dir, match_dir):
    coord_sections = set([os.path.basename(x).replace(".txt",'') for x in glob.glob(os.path.join(coord_dir, "*.txt")) ])
    match_sections = set([os.path.basename(x).replace(".h5",'') for x in glob.glob(os.path.join(match_dir, "*.h5")) ])
    to_do_coord_sections = coord_sections.difference(match_sections)
    if len(match_sections)==0:
        print(f"len(match_sections) {len(match_sections)}")
        print(os.path.join(coord_dir, "*.h5"))
    print(f"len(coord_sections) {len(coord_sections)} len(match_sections) {len(match_sections)} len(to_do_coord_sections) {len(to_do_coord_sections)}")
    if len(to_do_coord_sections)< len(coord_sections):
        print("reduced sections to stitch because", len(match_sections), "were already stitched")
    return sorted([os.path.join(coord_dir, x+".txt") for x in list(to_do_coord_sections)])

if __name__=='__main__':
    args = parse_args()
    root_dir, generate_settings, stitch_configs, num_cpus, mode, num_workers, nthreads, stitch_dir, coord_dir, mesh_dir, match_dir, render_meta_dir=setup_globals(args)
    print("stitch_mpi - num_workers",num_workers)
    print("work_dir", root_dir)
    if mode in ['matching', 'match']:
        if not RANK:
            print("mode", mode)
        print(f"stitch_mpi num_workers {stitch_configs['matching']['num_workers']}")
        coord_list = matching_check_status(coord_dir, match_dir)
        assert len(coord_list) > 0, f"didn't find any txt coord files {coord_dir}"
        sections_per_rank = int(math.ceil(len(coord_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(coord_list), 1)
        #print(RANK,"looking at indx", indx)
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_list, match_dir, **stitch_configs['matching'])
        time_region.log_summary()
    elif mode in ['rendering', 'render']:
        if not RANK:
            print("mode", mode)        
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
        #print(RANK,"looking at indx", indx)
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        stitch_configs_render.setdefault('meta_dir', render_meta_dir)
        print(f"image_outdir {image_outdir}")
        render_main(tform_list, image_outdir, **stitch_configs_render)   
        time_region.log_summary()     
    elif mode in ['optimization', 'optimize']:
        if not RANK:
            print("mode", mode)
        match_regex = os.path.abspath(os.path.join(match_dir, '*.h5'))
        match_list = sorted(glob.glob(match_regex))
        assert len(match_list) > 0, f"match list couldn't find any h5 files {match_regex}"
        sections_per_rank = int(math.ceil(len(match_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(match_list), 1)
        #print(RANK,"looking at indx", indx)
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        os.makedirs(mesh_dir, exist_ok=True)
        optmization_main(match_list, mesh_dir, **stitch_configs['optimization'])
        time_region.log_summary()
    elif mode == "matching_optimize":
        if not RANK:
            print("mode", mode)
        coord_list=sorted(glob.glob(os.path.join(coord_dir, "*.txt")))
        assert len(coord_list) > 0, f"didn't find any txt coord files {coord_dir}"
        sections_per_rank = int(math.ceil(len(coord_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(coord_list), 1)
        #print(RANK,"looking at indx", indx)
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_list, match_dir, **stitch_configs['matching'])
        time_region.log_summary()
        section_names = [os.path.basename(x).split(".")[0] for x in coord_list]
        match_list = sorted([os.path.join(match_dir, x+".h5") for x in section_names])
        os.makedirs(mesh_dir, exist_ok=True)
        optmization_main(match_list, mesh_dir, **stitch_configs['optimization'])
        time_region.log_summary()
    elif mode in ["matching_optimize_render", 'all'] :
        if not RANK:
            print("mode", mode)        
        coord_list=sorted(glob.glob(os.path.join(coord_dir, "*.txt")))
        assert len(coord_list) > 0, f"didn't find any txt coord files {coord_dir}"
        sections_per_rank = int(math.ceil(len(coord_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(coord_list), 1)
        #print(RANK,"looking at indx", indx)
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_list, match_dir, **stitch_configs['matching'])
        time_region.log_summary()
        section_names = sorted([os.path.basename(x).split(".")[0] for x in coord_list])
        match_list = [os.path.join(match_dir, x+".h5") for x in section_names]
        os.makedirs(mesh_dir, exist_ok=True)
        optmization_main(match_list, mesh_dir, **stitch_configs['optimization'])

        stitch_configs_render = stitch_configs['rendering']
        stitch_configs_render.pop('out_dir', '')
        image_outdir = config.stitch_render_dir(root_dir)
        driver = stitch_configs_render.get('driver', 'image')
        if driver == 'image':
            image_outdir = os.path.join(image_outdir, 'mip0')
        tform_list = [os.path.join(mesh_dir, x+".h5") for x in section_names]
        assert len(tform_list)>0, f"tform list empty, didn't find any h5 files"
        stitch_configs_render.setdefault('meta_dir', render_meta_dir)
        #print(f"image_outdir {image_outdir}")
        render_main(tform_list, image_outdir, **stitch_configs_render)   
        time_region.log_summary()