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
import numpy as np
import time
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()

if __name__=='__main__':
    args = parse_args()

    if args.mode=='meshing':
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        mesh_config = config_parts[2]
        if RANK==0:
            os.makedirs(align_config['mesh_dir'], exist_ok=True)
            secnames = sorted(glob.glob(os.path.join(align_config['thumbnail_dir'],'thumbnails', '*.png')))
            secnames = [os.path.basename(x).split(".")[0] for x in secnames]
            #match_list = sorted(glob.glob(os.path.join(align_config['thumb_match_dir'], '*.h5')))
            if args.reverse:
                secnames = secnames[::-1]
                #match_list = match_list[::-1]
            #match_list= np.array_split(np.array(match_list), NUMRANKS,axis=0)
            secnames = np.array_split(np.array(secnames), NUMRANKS,axis=0)
        else:
            #match_list=None
            secnames=None
        secnames = comm.scatter(secnames, root=0)
        #match_list = comm.scatter(match_list, root=0)
        #for i in range(NUMRANKS):
        #    if i==RANK:
        #        print("rank", i, secnames)
        generate_mesh_main(align_config,mesh_config,match_list=False ,secnames=secnames)
        if RANK==0:
            print(os.listdir(align_config['mesh_dir']))
        comm.barrier()
    elif args.mode == 'optimization':
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        optimization_config = config_parts[2]
        if RANK==0:
            if not os.path.exists(align_config['tform_dir']):
                os.mkdir(align_config['tform_dir'])
            print("Warning: only one rank is doing the optimization")
            optimize_main(None, align_config, optimization_config)
        comm.barrier()
    elif args.mode=='render':
        start_align_render_configs =time.time() 
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        render_config = config_parts[2]
        comm.barrier()
        time_region.track_time('align_mpi.align_render_configs', time.time() - start_align_render_configs)
        if RANK==0:
            os.makedirs(align_config['render_dir'], exist_ok=True)
            tform_list_all =sorted( glob.glob(os.path.join(align_config['tform_dir'],"*.h5")))
            tform_list = np.array_split(np.array(tform_list_all), NUMRANKS,axis=0)
            print("align_config['work_dir']",align_config['work_dir'])
            print("tform_list", tform_list)
            z_prefix = {}
            if align_config.pop('prefix_z_number', True):
                seclist = sorted(glob.glob(os.path.join(align_config['mesh_dir'], '*.h5')))
                section_order_file = os.path.join(align_config['work_dir'], 'section_order.txt')
                seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
                zero_padding = math.ceil(math.log10(len(seclist)))
                z_prefix.update({os.path.basename(s): str(k).zfill(zero_padding)+'_'
                                for k, s in zip(z_indx, seclist)})
            
        else:
            tform_list_all=None
            tform_list = None
            z_prefix=None
        ##these two lines are basically a scatter operation, but scatter wasn't sending the data to the other ranks. rank 0 would get what it
        ##   needed but rank 1 would get None
        tform_list = comm.bcast(tform_list , 0)
        tform_list = tform_list[RANK]
        z_prefix = comm.bcast(z_prefix,0)
        #for i in range(NUMRANKS):
        #    if i==RANK:
        #        print("rank", RANK, "tform_list", tform_list)        
        #print("align_mpi stitch_render_config['out_dir']",stitch_render_config['out_dir'])
        if tform_list is None:
            raise Exception("tform_list shouldn't be None. Are there more ranks than sections?")  
   
        start_stitch_render_configs=time.time() 
        ##stitch_render_config is being split up with bcase however render_config is loaded from file by every rank
        ##    there is no reason for these two to be different. 
        if RANK==0:
            stitch_render_config = config.stitch_configs(align_config['work_dir']).get('rendering', {})
            if stitch_render_config['out_dir'] is None:
                stitch_render_config['out_dir'] = os.path.join(align_config['work_dir'], 'stitched_sections')
        else:
            stitch_render_config=None
        stitch_render_config=comm.bcast(stitch_render_config,0)
        time_region.track_time('align_mpi.stitch_render_configs', time.time() - start_stitch_render_configs)
        start_rendering_offset =time.time() 
        if RANK==0:
            if align_config.pop('offset_bbox', True):
                offset_name = os.path.join(align_config['tform_dir'], 'offset.txt')
                if not os.path.isfile(offset_name):
                    time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                    offset_bbox_main(align_config, tform_list_all)
        time_region.track_time('align_mpi.stitch_render_offset', time.time() - start_rendering_offset)
        
        #print("align_mpi render_config['render_dir']",render_config['render_dir'])
        comm.barrier()
        start_render_main = time.time() 
        render_main(tform_list, render_config, align_config['tform_dir'], z_prefix, stitch_render_config=stitch_render_config)
        time_region.track_time('align_mpi.render_main', time.time() - start_render_main)
        comm.barrier()

        time_region.log_summary()        
    elif args.mode=='pipeline':
        raise Exception("the scattering in here is not working. non-rank 0 gets None instead of data.")
        args.mode = 'meshing'
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        mesh_config = config_parts[2]
        os.makedirs(align_config['mesh_dir'], exist_ok=True)
        if RANK==0:
            match_list = sorted(glob.glob(os.path.join(align_config['thumb_match_dir'], '*.h5')))
            if args.reverse:
                match_list = match_list[::-1]
            match_list= np.array_split(np.array(match_list), NUMRANKS,axis=0)
        else:
            match_list=None
        match_list = comm.scatter(match_list, root=0)
    
        generate_mesh_main(align_config,mesh_config,match_list=match_list )
        if RANK==0:
            print(os.listdir(align_config['mesh_dir']))
        comm.barrier()
        
        args.mode = 'optimization'
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        optimization_config = config_parts[2]
        if RANK==0:
            if not os.path.exists(align_config['tform_dir']):
                os.mkdir(align_config['tform_dir'])
            print("Warning: only one rank is doing the optimization")
            optimize_main(None, align_config, optimization_config)
        comm.barrier()
        args.mode = 'render'
        start_align_render_configs =time.time() 
        config_parts = setup_configs(args)
        align_config = config_parts[0]
        mode = config_parts[1]
        render_config = config_parts[2]
        comm.barrier()
        time_region.track_time('align_mpi.align_render_configs', time.time() - start_align_render_configs)
        start_stitch_render_configs=time.time() 
        if RANK==0:
            tform_list_all =sorted( glob.glob(os.path.join(align_config['tform_dir'],"*.h5")))
            tform_list = np.array_split(np.array(tform_list_all), NUMRANKS,axis=0)
            print("align_config['work_dir']",align_config['work_dir'])
            print("tform_list", tform_list)
            stitch_render_config = config.stitch_configs(align_config['work_dir']).get('rendering', {})
            if stitch_render_config['out_dir'] is None:
                stitch_render_config['out_dir'] = os.path.join(align_config['work_dir'], 'stitched_sections')
        else:
            stitch_render_config=None
            tform_list = None
        tform_list = comm.bcast(tform_list , 0)
        tform_list = tform_list[RANK]
        stitch_render_config=comm.bcast(stitch_render_config,0)
        for i in range(NUMRANKS):
            if i==RANK:
                print("rank", RANK, "tform_list", tform_list)        
        #print("align_mpi stitch_render_config['out_dir']",stitch_render_config['out_dir'])
        if tform_list is None:
            raise Exception("tform_list shouldn't be None. Are there more ranks than sections?")        
        time_region.track_time('align_mpi.stitch_render_configs', time.time() - start_stitch_render_configs)
        start_rendering =time.time() 
        if align_config.pop('offset_bbox', True):
            offset_name = os.path.join(align_config['tform_dir'], 'offset.txt')
            if not os.path.isfile(offset_name):
                time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                offset_bbox_main(align_config, tform_list)
        os.makedirs(align_config['render_dir'], exist_ok=True)
        #tform_list = sorted(glob.glob(os.path.join(align_config['tform_dir'], '*.h5')))
        
        #tform_list = tform_list[indx]
        #assert len(match_list)==len(tform_list),f"len(match_list) {len(match_list)}  len(tform_list)  {len(tform_list)}"
        z_prefix = defaultdict(lambda: '')
        if align_config.pop('prefix_z_number', True):
            seclist = sorted(glob.glob(os.path.join(align_config['mesh_dir'], '*.h5')))
            section_order_file = os.path.join(align_config['work_dir'], 'section_order.txt')
            seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
            digit_num = math.ceil(math.log10(len(seclist)))
            z_prefix.update({os.path.basename(s): str(k).rjust(digit_num, '0')+'_'
                             for k, s in zip(z_indx, seclist)})
        #print("align_mpi render_config['render_dir']",render_config['render_dir'])
        render_main(tform_list, render_config, align_config['tform_dir'], z_prefix, stitch_render_config=stitch_render_config)
        comm.barrier()
        time_region.track_time('align_mpi.rendering', time.time() - start_rendering)
        time_region.log_summary()