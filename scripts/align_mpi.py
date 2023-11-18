from feabas import common
import argparse
from align_main import parse_args, generate_mesh_main, optimize_main
from feabas.time_region import time_region
from feabas import config, logging
from mpi4py import MPI
import math
import os
from collections import defaultdict
from align_main import setup_configs, render_main,offset_bbox_main
import glob
import time
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()

if __name__=='__main__':
    args = parse_args()

    if args.mode=='pipeline':
        args.mode = 'meshing'
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        mesh_config = config_parts[2]
        os.makedirs(align_config['mesh_dir'], exist_ok=True)
        match_list = glob.glob(os.path.join(align_config['thumb_match_dir'], '*.h5'))
        sections_per_rank = int(math.ceil(len(match_list)/NUMRANKS))
        if RANK!=(NUMRANKS-1):
            indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
        else:
            indx = slice(RANK*sections_per_rank, len(match_list), 1)
        match_list = match_list[indx]
        generate_mesh_main(align_config,mesh_config,match_list=match_list )
        comm.barrier()

        args.mode = 'optimization'
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        optimization_config = config_parts[2]
        if RANK==0:
            print("only one rank is doing the optimization")
            optimize_main(None, align_config, optimization_config)
        comm.barrier()

        args.mode = 'render'
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        render_config = config_parts[2]
        start_rendering =time.time() 
        if align_config.pop('offset_bbox', True):
            offset_name = os.path.join(align_config['tform_dir'], 'offset.txt')
            if not os.path.isfile(offset_name):
                time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                offset_bbox_main(align_config)
        os.makedirs(align_config['render_dir'], exist_ok=True)
        tform_list = sorted(glob.glob(os.path.join(align_config['tform_dir'], '*.h5')))
        tform_list = tform_list[indx]
        assert len(match_list)==len(tform_list),f"len(match_list) {len(match_list)}  len(tform_list)  {len(tform_list)}"
        if args.reverse:
            tform_list = tform_list[::-1]
        z_prefix = defaultdict(lambda: '')
        if align_config.pop('prefix_z_number', True):
            seclist = sorted(glob.glob(os.path.join(align_config['mesh_dir'], '*.h5')))
            section_order_file = os.path.join(align_config['work_dir'], 'section_order.txt')
            seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
            digit_num = math.ceil(math.log10(len(seclist)))
            z_prefix.update({os.path.basename(s): str(k).rjust(digit_num, '0')+'_'
                             for k, s in zip(z_indx, seclist)})
        render_main(tform_list, render_config, align_config['tform_dir'], z_prefix)
        comm.barrier()
        time_region.track_time('align_main.rendering', time.time() - start_rendering)
