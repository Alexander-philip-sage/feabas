import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
import math
from multiprocessing import get_context
import os, sys
import time
from typing import List
from feabas.time_region import time_region
from feabas import config, logging
from feabas import mipmap, common, material
from feabas.mipmap import mip_map_one_section
from thumbnail_main import setup_globals, setup_pairnames, setup_bnames, parse_args, downsample_main, setup_pair_names, align_main
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
NUMRANKS = comm.Get_size()

def setup_neuroglancer_precomputed(work_dir):
    stitch_dir = os.path.join(work_dir, 'stitch')
    meta_dir = os.path.join(stitch_dir, 'ts_specs')
    meta_regex = os.path.join(meta_dir,'*.json')
    meta_list = sorted(glob.glob(meta_regex))
    assert len(meta_list) > 0, f"did not find any json files in {os.path.abspath(meta_regex)}"
    sections_per_rank = int(math.ceil(len(meta_list)/NUMRANKS))
    if RANK!=(NUMRANKS-1):
        arg_indx = slice(RANK*sections_per_rank, (RANK+1)*sections_per_rank, 1)
    else:
        arg_indx = slice(RANK*sections_per_rank, len(meta_list), 1)
    return meta_list[arg_indx], meta_dir

def mpi_downsample(thumbnail_configs,stitch_render_conf,work_dir ):
    min_mip = thumbnail_configs.get('min_mip', 0)
    stitched_dir = stitch_render_conf['out_dir']
    meta_dir = os.path.join(stitched_dir, 'mip'+str(min_mip), '**', 'metadata.txt')
    if RANK==0:
        meta_list = sorted(glob.glob(meta_dir, recursive=True))
        #print("meta_list", meta_list)
        meta_list= np.array_split(np.array(meta_list), NUMRANKS,axis=0)
    else:
        meta_list=None
    meta_list = comm.scatter(meta_list, root=0)
    assert len(meta_list)>0, f"did not find any metadata.txt files in {os.path.abspath(meta_dir)}"
    if  RANK==0:
        print(f"after scatter len(meta_list) {len(meta_list)}")
    if RANK==0:
        sanity_test = [1,2,3]
        sanity_test2 = np.array([1,2,3])
    else:
        sanity_test = None
        sanity_test2 = None
    sanity_test = comm.bcast(sanity_test, root=0)
    #sanity_test2 = comm.Bcast(sanity_test2, root=0)
    for i in range(NUMRANKS):
        if i==RANK:
            print("mpid rank", RANK,"sanity test", sanity_test)
            print("mpid rank", RANK, "sanity_test2", sanity_test2)
    downsample_main(thumbnail_configs,stitch_render_conf=stitch_render_conf, work_dir=work_dir,meta_list = meta_list)
    if RANK==0:
        sanity_test = [1,2,3]
        sanity_test2 = np.array([1,2,3])
    else:
        sanity_test = None
        sanity_test2 = None
    sanity_test = comm.bcast(sanity_test, root=0)
    #sanity_test2 = comm.Bcast(sanity_test2, root=0)
    for i in range(NUMRANKS):
        if i==RANK:
            print("mpiafd rank", RANK,"sanity test", sanity_test)
            print("mpiafd rank", RANK, "sanity_test2", sanity_test2)    
    comm.barrier()
    print("rank", RANK, "finished mpi_downsample") 

def mpi_alignment(thumbnail_configs,num_workers, thumbnail_img_dir):
    compare_distance = thumbnail_configs.pop('compare_distance', 1)

    if RANK==0:
        print("work_dir", work_dir)
        _, bname_list = setup_bnames(thumbnail_img_dir, work_dir)
        bname_list_np = np.array(bname_list)
        bname_blist = [w.encode("utf-8") for w in bname_list]
        bname_bstr = ','.join(bname_list).encode("utf-8")
        bname_str = ','.join(bname_list)
        print("bname_list", type(bname_list), bname_list)
        sanity_test = [1,2,3]
        sanity_test2 = np.array([1,2,3])
    else:
        bname_bstr=None
        bname_blist= None
        bname_str=None
        bname_list=None
        bname_list_np=None
        sanity_test = None
        sanity_test2 = None
    #bname_list = comm.Bcast(bname_list_np, root=0)
    #bname_blist = comm.Bcast(bname_blist, root=0)
    #bname_str = comm.bcast(bname_str, root=0)
    #bname_bstr = comm.bcast(bname_bstr, root=0)
    #bname_str = comm.Bcast([bname_str, MPI.CHAR], root=0)
    #bname_bstr = comm.Bcast([bname_bstr, MPI.CHAR], root=0)
    #bname_list = bname_str.split(',')  
    #bname_list = bname_bstr.decode("utf-8").split(',')  
    #bname_list = [w.encode("utf-8") for w in bname_blist]
    #bname_list = comm.Bcast(bname_list_np, root=0)
    #bname_list = comm.bcast(bname_list, root=0)

    sanity_test = comm.bcast(sanity_test, root=0)
    #sanity_test2 = comm.Bcast(sanity_test2, root=0)
    for i in range(NUMRANKS):
        if i==RANK:
            print("mpia rank", RANK,"sanity test", sanity_test)
            print("mpia rank", RANK, "sanity_test2", sanity_test2)
    return
    if bname_list is None or len(bname_list)==0:
        print(f"rank {RANK}: bname list must be greater than 0" , file=sys.stderr)
        #comm.Abort( )   
    pairnames = setup_pairnames(bname_list, compare_distance)
    pairnames = np.array_split(np.array(pairnames), NUMRANKS, axis=0)[RANK]
    if pairnames is None:
        print("pairnames shouldn't be None. Are there more ranks than section pairs?", file=sys.stderr)
        #comm.Abort()
    else:
        print("rank", RANK, "pairnames", pairnames)
    align_main(thumbnail_configs,pairnames=pairnames, num_workers=num_workers)
    print("rank", RANK, "before comm barrier")
    comm.barrier()
    print("rank", RANK, "after comm barrier")
    time_region.log_summary()

if __name__=='__main__':
    args = parse_args()
    num_workers=None
    arg_indx=None
    if args.mode=='downsample':
        if RANK==0:
            args.mode = 'downsample'
            _, thumbnail_configs, _, stitch_render_conf= setup_globals(args)
            driver = stitch_render_conf.get('driver', 'image')
            assert driver=='image'
        else:
            thumbnail_configs = None
            stitch_render_conf=None
        stitch_render_conf=comm.bcast(stitch_render_conf, root=0)
        thumbnail_configs = comm.bcast(thumbnail_configs, root=0)
        work_dir = thumbnail_configs['work_dir']
        if RANK==0:
            print("work_dir", work_dir)
        stitch_conf = config.stitch_configs(work_dir)['rendering']
        driver = stitch_conf.get('driver', 'image')
        if driver == 'image':      
            mpi_downsample(thumbnail_configs,work_dir )  
        elif driver =='neuroglancer_precomputed':
            meta_list, meta_dir = setup_neuroglancer_precomputed(work_dir)
            downsample_main(thumbnail_configs,meta_list=meta_list)     
        comm.barrier()   
    elif args.mode == 'alignment':
        if RANK==0:
            args.mode = 'downsample'
            _, thumbnail_configs, _, stitch_render_conf= setup_globals(args)
            driver = stitch_render_conf.get('driver', 'image')
            assert driver=='image'
        else:
            thumbnail_configs = None
            stitch_render_conf=None
        stitch_render_conf=comm.bcast(stitch_render_conf, root=0)
        thumbnail_configs = comm.bcast(thumbnail_configs, root=0)
        thumbnail_img_dir=thumbnail_configs['thumbnail_img_dir']
        mpi_alignment(thumbnail_configs,num_workers, thumbnail_img_dir)
    elif args.mode =='downsample_precomputed_alignment':
        args.mode = 'downsample'
        general_settings, thumbnail_configs, mode, stitch_render_conf= setup_globals(args)
        work_dir = thumbnail_configs['work_dir']
        thumbnail_img_dir=thumbnail_configs['thumbnail_img_dir']
        if RANK==0:
            print("work_dir", work_dir)
        stitch_conf = config.stitch_configs(work_dir)['rendering']
        driver = stitch_conf.get('driver', 'image')
        meta_list, meta_dir = setup_neuroglancer_precomputed(work_dir)
        section_names = sorted([os.path.basename(x).split(".")[0] for x in meta_list])
        downsample_main(thumbnail_configs, meta_list=meta_list)     
        args.mode = 'alignment'
        general_settings, thumbnail_configs, mode, stitch_render_conf= setup_globals(args)
        compare_distance = thumbnail_configs.pop('compare_distance', 1)
        imglist = [os.path.join(thumbnail_img_dir, x+".png") for x in section_names]
        _, bname_list, pairnames = setup_pair_names(thumbnail_img_dir,work_dir,  compare_distance, imglist=imglist)
        align_main(thumbnail_configs,pairnames=pairnames, num_workers=num_workers)
        comm.barrier()
        time_region.log_summary()
    elif args.mode=='downsample_alignment':
        #raise Exception("this only works on one rank for unknown reasons. call downsample and alignment seperately. fails in mpi_alignment's scatter. rank 1 gets None instead of data")
        
        if RANK==0:
            args.mode = 'downsample'
            _, thumbnail_configs_d, _, stitch_render_conf= setup_globals(args)
            driver = stitch_render_conf.get('driver', 'image')
            assert driver=='image'
            args.mode = 'alignment'
            _, thumbnail_configs_a, _, _= setup_globals(args)
        else:
            thumbnail_configs_a = None
            thumbnail_configs_d = None
            stitch_render_conf=None
        stitch_render_conf=comm.bcast(stitch_render_conf, root=0)
        thumbnail_configs_a = comm.bcast(thumbnail_configs_a, root=0)
        thumbnail_configs_d = comm.bcast(thumbnail_configs_d, root=0)
        work_dir = thumbnail_configs_a['work_dir']
        if RANK==0:
            print("work_dir", work_dir)

        mpi_downsample(thumbnail_configs_a, stitch_render_conf,work_dir )
        if RANK==0:
            sanity_test = [1,2,3]
            sanity_test2 = np.array([1,2,3])
        else:
            sanity_test = None
            sanity_test2 = None
        sanity_test = comm.bcast(sanity_test, root=0)
        #sanity_test2 = comm.Bcast(sanity_test2, root=0)
        for i in range(NUMRANKS):
            if i==RANK:
                print("mpim rank", RANK,"sanity test", sanity_test)
                print("mpim rank", RANK, "sanity_test2", sanity_test2)
        thumbnail_img_dir=thumbnail_configs_a['thumbnail_img_dir']
        mpi_alignment(thumbnail_configs_a,num_workers, thumbnail_img_dir)
        print("finished all downsample_alignment")
