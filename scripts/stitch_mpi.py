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
from stitch_main import parse_args, match_main, setup_globals
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
        coord_list = coord_list[indx]    
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        match_main(coord_list, match_dir, **stitch_configs)
    elif