from collections import defaultdict
import argparse
import glob
from functools import partial
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import get_context
import math
import os
import time
import gc
from typing import List
from feabas.time_region import time_region
from feabas import config, logging
import feabas.constant as const
from feabas import material, dal, common
from feabas.mesh import Mesh
from feabas.mipmap import get_image_loader, mip_map_one_section
from feabas.aligner import match_section_from_initial_matches
from feabas.renderer import render_whole_mesh, VolumeRenderer
import numpy as np


def generate_mesh_from_mask(mask_names, outname, **kwargs):
    if os.path.isfile(outname):
        return
    from feabas import material, dal, spatial, mesh
    material_table = kwargs.get('material_table', material.MaterialTable())
    target_resolution = kwargs.get('target_resolution', config.DEFAULT_RESOLUTION)
    mesh_size = kwargs.get('mesh_size', 600)
    simplify_tol = kwargs.get('simplify_tol', 2)
    area_thresh = kwargs.get('area_thresh', 0)
    logger_info = kwargs.pop('logger', None)
    #logger = logging.get_logger(logger_info)
    if isinstance(simplify_tol, dict):
        region_tols = defaultdict(lambda: 0.1)
        region_tols.update(simplify_tol)
    else:
        region_tols = defaultdict(lambda: simplify_tol)
    loader = None
    if not isinstance(material_table, material.MaterialTable):
        if isinstance(material_table, dict):
            material_table = material.MaterialTable(table=material_table)
        elif isinstance(material_table, str):
            material_table = material.MaterialTable.from_json(material_table, stream=not material_table.endswith('.json'))
        else:
            raise TypeError
    if 'exclude' in material_table.named_table:
        mat = material_table['exclude']
        fillval = mat.mask_label
    else:
        fillval = 255
    for mask_name, resolution in mask_names:
        if not os.path.isfile(mask_name):
            continue
        src_resolution = resolution
        if mask_name.lower().endswith('.json') or mask_name.lower().endswith('.txt'):
            loader = dal.get_loader_from_json(mask_name, resolution=src_resolution, fillval=fillval)
        else:
            loader = mask_name
        break
    secname = os.path.splitext(os.path.basename(outname))[0]
    if loader is None:
        #logger.warning(str(secname) + " mask does not exist")
        return
    mesh_size = mesh_size * config.DEFAULT_RESOLUTION / src_resolution
    G = spatial.Geometry.from_image_mosaic(loader, material_table=material_table, resolution=src_resolution)
    PSLG = G.PSLG(region_tol=region_tols,  roi_tol=0, area_thresh=area_thresh)
    M = mesh.Mesh.from_PSLG(**PSLG, material_table=material_table, mesh_size=mesh_size, min_mesh_angle=20)
    M.change_resolution(target_resolution)
    if ('split' in material_table.named_table):
        mid = material_table.named_table['split'].uid
        m_indx = M.material_ids == mid
        M.incise_region(m_indx)
    mshname = os.path.splitext(os.path.basename(mask_name))[0]
    M.save_to_h5(outname, save_material=True, override_dict={'name': mshname})


def generate_mesh_main(align_config: dict, mesh_config: dict, match_list: List[str]=None, secnames: List[str] =None):
    num_workers = mesh_config['num_workers']
    start_generate_mesh_main=time.time()
    #logger_info = logging.initialize_main_logger(logger_name='mesh_generation', mp=num_workers>1)
    #mesh_config['logger'] = logger_info[0]
    #logger = logging.get_logger(logger_info[0])
    thumbnail_mip_lvl = align_config.get('thumbnail_mip_level', 6)
    thumbnail_resolution = config.DEFAULT_RESOLUTION * (2 ** thumbnail_mip_lvl)
    thumbnail_mask_dir = os.path.join(align_config['thumbnail_dir'], 'material_masks')
    if  match_list is None:
        print("file lookup with glob")
        match_list = glob.glob(os.path.join(align_config['thumb_match_dir'], '*.h5'))
        assert len(match_list)>0, f"must find more than one match in {os.path.abspath(align_config['thumb_match_dir'])}"
    if secnames is None:
        match_names = [os.path.basename(s).replace('.h5', '').split(align_config['match_name_delimiter']) for s in match_list]
        secnames = set([s for pp in match_names for s in pp])
    alt_mask_dir = mesh_config.get('mask_dir', None)
    alt_mask_mip_level = mesh_config.get('mask_mip_level', 4)
    alt_mask_resolution = config.DEFAULT_RESOLUTION * (2 ** alt_mask_mip_level)
    if alt_mask_dir is None:
        alt_mask_dir = os.path.join(align_config['align_dir'], 'material_masks')
    material_table_file = config.material_table_file(align_config.get("work_dir", None))
    material_table = material.MaterialTable.from_json(material_table_file, stream=False)
    if num_workers == 1:
        for sname in secnames:
            mask_names = [(os.path.join(alt_mask_dir, sname + '.json'), alt_mask_resolution),
                        (os.path.join(alt_mask_dir, sname + '.txt'), alt_mask_resolution),
                        (os.path.join(alt_mask_dir, sname + '.png'), alt_mask_resolution),
                        (os.path.join(thumbnail_mask_dir, sname + '.png'), thumbnail_resolution)]
            outname = os.path.join(align_config['mesh_dir'], sname + '.h5')
            generate_mesh_from_mask(mask_names, outname, material_table=material_table, **mesh_config)
    else:
        material_table = material_table.save_to_json(jsonname=None)
        target_func = partial(generate_mesh_from_mask, material_table=material_table, **mesh_config)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for sname in secnames:
                mask_names = [(os.path.join(alt_mask_dir, sname + '.json'), alt_mask_resolution),
                              (os.path.join(alt_mask_dir, sname + '.txt'), alt_mask_resolution),
                              (os.path.join(alt_mask_dir, sname + '.png'), alt_mask_resolution),
                              (os.path.join(thumbnail_mask_dir, sname + '.png'), thumbnail_resolution)]
                outname = os.path.join(align_config['mesh_dir'], sname + '.h5')
                if not os.path.isfile(outname):
                    job = executor.submit(target_func, mask_names=mask_names, outname=outname)
                    jobs.append(job)
            for job in jobs:
                job.result()
    time_region.track_time('align_main.generate_mesh_main', time.time() - start_generate_mesh_main)                
    #logger.info('meshes generated.')
    #logging.terminate_logger(*logger_info)


def match_main(align_config,match_config, match_list):
    stitch_config = config.stitch_configs(align_config['work_dir']).get('rendering', {})
    loader_config = {key: val for key, val in stitch_config.items() if key in ('pattern', 'one_based', 'fillval')}
    working_mip_level = align_config.get('working_mip_level', 2)
    stitch_render_driver = config.stitch_configs().get('rendering', {}).get('driver', 'image')
    if stitch_render_driver == 'image':
        stitch_render_dir = config.stitch_render_dir()
        stitched_image_dir = os.path.join(stitch_render_dir, 'mip'+str(working_mip_level))
    else:
        stitch_dir = os.path.join(align_config['work_dir'], 'stitch')
        spec_dir = os.path.join(stitch_dir, 'ts_specs')
    logger_info = logging.initialize_main_logger(logger_name='align_matching', mp=False)
    logger = logging.get_logger(logger_info[0])
    if len(match_list) == 0:
        return
    for mname in match_list:
        outname = os.path.join(align_config['match_dir'], os.path.basename(mname))
        if os.path.isfile(outname):
            continue
        t0 = time.time()
        tname = os.path.basename(mname).replace('.h5', '')
        logger.info(f'start {tname}')
        secnames = os.path.splitext(os.path.basename(mname))[0].split(align_config['match_name_delimiter'])
        if stitch_render_driver == 'image':
            loaders = [get_image_loader(os.path.join(stitched_image_dir, s), **loader_config) for s in secnames]
        else:
            specs = [dal.get_tensorstore_spec(os.path.join(spec_dir, s+'.json'), mip=working_mip_level) for s in secnames]
            loader0 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[0]}
            loader1 = {'ImageLoaderType': 'TensorStoreLoader', 'json_spec': specs[1]}
            loader0.update(loader_config)
            loader1.update(loader_config)
            loaders = [loader0, loader1]
        num_matches = match_section_from_initial_matches(mname, align_config['mesh_dir'], loaders, align_config['match_dir'], match_config)
        if num_matches is not None:
            if num_matches > 0:
                logger.info(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
            else:
                logger.warning(f'{tname}: {num_matches} matches, {round((time.time()-t0)/60,3)} min.')
        gc.collect()
    logger.info('matching finished.')
    logging.terminate_logger(*logger_info)


def optimize_main(section_list,align_config,  optimization_config):
    start_optimize_main=time.time() 
    from feabas.aligner import Stack
    stack_config = optimization_config.get('stack_config', {}).copy()
    slide_window = optimization_config.get('slide_window', {}).copy()
    logger_info = logging.initialize_main_logger(logger_name='align_optimization', mp=optimization_config['slide_window']['num_workers']>1)
    stack_config.setdefault('section_order_file', os.path.join(align_config['work_dir'], 'section_order.txt'))
    slide_window['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    #print("optimize_main: section_list before Stack object created", section_list)
    stk = Stack(section_list=section_list, mesh_dir=align_config['mesh_dir'], match_dir=align_config['match_dir'], 
                mesh_out_dir=align_config['tform_dir'], **stack_config)
    section_list = stk.section_list
    #print("optimize_main: section_list after Stack object created", section_list)
    stk.update_lock_flags({s: os.path.isfile(os.path.join(align_config['tform_dir'], s + '.h5')) for s in section_list})
    locked_flags = stk.locked_array
    logger.info(f'{locked_flags.size} images| {np.sum(locked_flags)} references')
    cost = stk.optimize_slide_window(optimize_rigid=True, optimize_elastic=True,
        target_gear=const.MESH_GEAR_MOVING, **slide_window)
    if os.path.isfile(os.path.join(align_config['tform_dir'], 'residue.csv')):
        cost0 = {}
        with open(os.path.join(align_config['tform_dir'], 'residue.csv'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                mn, dis0, dis1 = line.split(', ')
                cost0[mn] = (float(dis0), float(dis1))
        cost0.update(cost)
        cost = cost0
    with open(os.path.join(align_config['tform_dir'], 'residue.csv'), 'w') as f:
        mnames = sorted(list(cost.keys()))
        for key in mnames:
            val = cost[key]
            f.write(f'{key}, {val[0]}, {val[1]}\n')
    time_region.track_time('align_main.optimize_main', time.time() - start_optimize_main)
    #logger.info('finished')
    logging.terminate_logger(*logger_info)

def offset_bbox_main(align_config,tform_list: List[str]=None):
    logger_info = logging.initialize_main_logger(logger_name='offset_bbox', mp=False)
    logger = logging.get_logger(logger_info[0])
    outname = os.path.join(align_config['tform_dir'], 'offset.txt')
    if tform_list is None:
        print("file lookup with glob")
        tform_list = sorted(glob.glob(os.path.join(align_config['tform_dir'], '*.h5')))
    if os.path.isfile(outname) or (len(tform_list) == 0):
        return
    secnames = [os.path.splitext(os.path.basename(s))[0] for s in tform_list]
    mip_level = align_config.pop('get', 0)
    outdir = os.path.join(align_config['render_dir'], 'mip'+str(mip_level))
    for sname in secnames:
        if os.path.isdir(os.path.join(outdir, sname)):
            logger.info(f'section {sname} already rendered: transformation not performed')
            return
    bbox_union = None
    for tname in tform_list:
        M = Mesh.from_h5(tname)
        M.change_resolution(config.DEFAULT_RESOLUTION)
        bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
        if bbox_union is None:
            bbox_union = bbox
        else:
            bbox_union = common.bbox_union((bbox_union, bbox))
    offset = -bbox_union[:2]
    bbox_union_new = bbox_union + np.tile(offset, 2)
    if not os.path.isfile(outname):
        with open(outname, 'w') as f:
            f.write('\t'.join([str(s) for s in offset]))
    logger.warning(f'bbox offset: {tuple(bbox_union)} -> {tuple(bbox_union_new)}')
    logging.terminate_logger(*logger_info)


def render_one_section(h5name, z_prefix='', stitch_render_config:dict =None, **kwargs):
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    mip_level = kwargs.pop('mip_level', 0)
    offset = kwargs.pop('offset', None)
    render_dir = kwargs.pop('render_dir', None)
    
    work_dir = kwargs.pop('work_dir', None)
    secname = os.path.splitext(os.path.basename(h5name))[0]
    outdir = os.path.join(render_dir, 'mip'+str(mip_level), z_prefix+secname)
    resolution = config.DEFAULT_RESOLUTION * (2 ** mip_level)
    meta_name = os.path.join(outdir, 'metadata.txt')
    if os.path.isfile(meta_name):
        return None
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    if stitch_render_config is None:
        stitch_render_config = config.stitch_configs(work_dir).get('rendering', {})
        stitch_render_dir = config.stitch_render_dir()
    else:
        stitch_render_dir =stitch_render_config['out_dir']
        #print("render_one_section stitch_render_dir",stitch_render_dir)
    loader_config = kwargs.pop('loader_config', {}).copy()
    loader_config.update({key: val for key, val in stitch_render_config.items() if key in ('pattern', 'one_based', 'fillval')})
    
    stitched_image_dir = os.path.join(stitch_render_dir, 'mip'+str(mip_level))
    loader_config['resolution'] = resolution
    loader = get_image_loader(os.path.join(stitched_image_dir, secname), **loader_config)
    #print("render_one_section h5name", h5name)
    M = Mesh.from_h5(h5name)
    M.change_resolution(resolution)
    if offset is not None:
        M.apply_translation(offset * config.DEFAULT_RESOLUTION/resolution, gear=const.MESH_GEAR_MOVING)
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, secname)
    #print("prefix for render_whole mesh", prefix)
    rendered = render_whole_mesh(M, loader, prefix, **kwargs)
    fnames = sorted(list(rendered.keys()))
    bboxes = []
    for fname in fnames:
        bboxes.append(rendered[fname])
    out_loader = dal.StaticImageLoader(fnames, bboxes=bboxes, resolution=resolution)
    out_loader.to_coordinate_file(meta_name)
    logger.info(f'{secname}: {len(rendered)} tiles | {time.time()-t0} secs.')
    return len(rendered)


def render_main(tform_list, render_config, tform_dir, z_prefix=None, stitch_render_config:dict =None):
    logger_info = logging.initialize_main_logger(logger_name='align_render', mp=False)
    #print("render_main stitch_render_config['out_dir']",stitch_render_config['out_dir'])
    render_config['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    num_workers = render_config.get('num_workers', 1)
    cache_size = render_config.get('loader_config', {}).get('cache_size', None)
    #print("render_main cache_size", cache_size)
    if (cache_size is not None) and (num_workers > 1):
        render_config.setdefault('loader_config', {})
        render_config['loader_config'].setdefault('cache_size', cache_size // num_workers)
    offset_name = os.path.join(tform_dir, 'offset.txt')
    if os.path.isfile(offset_name):
        with open(offset_name, 'r') as f:
            line = f.readline()
        offset = np.array([float(s) for s in line.strip().split('\t')])
        logger.info(f'use offset {offset}')
    else:
        offset = None
    if z_prefix is None:
        z_prefix = {}
    for tname in tform_list:
        z = z_prefix.get(os.path.basename(tname), '')
        render_one_section(tname, z_prefix=z, offset=offset, stitch_render_config=stitch_render_config, **render_config)
    #logger.info('finished')
    logging.terminate_logger(*logger_info)


def generate_aligned_mipmaps(render_dir, max_mip, meta_list=None, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = logging.initialize_main_logger(logger_name='align_mipmap', mp=num_workers>0)
    kwargs['logger'] = logger_info[0]
    logger = logging.get_logger(logger_info[0])
    if meta_list is None:
        print("file lookup with glob")
        meta_list = sorted(glob.glob(os.path.join(render_dir, 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            mip_map_one_section(sname, render_dir, max_mip, num_workers=num_workers, **kwargs)
    else:
        target_func = partial(mip_map_one_section, img_dir=render_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for sname in secnames:
                job = executor.submit(target_func, sname)
                jobs.append(job)
            for job in jobs:
                job.result()
    #logger.info('mipmapping generated.')
    logging.terminate_logger(*logger_info)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run alignment")
    parser.add_argument("--work_dir", metavar="work_dir", type=str, default=None)
    parser.add_argument("--mode", metavar="mode", type=str, default='matching')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)

def setup_configs(args):
    if not args.work_dir:
        work_dir = config.get_work_dir()
        generate_settings = config.general_settings()
        align_config = config.align_configs()
    else:
        work_dir = args.work_dir
        config._default_configuration_folder = os.path.join(args.work_dir, 'configs')
        generate_settings = config.general_settings(config._default_configuration_folder)
        align_config = config.align_configs(work_dir)

    num_cpus = generate_settings['cpu_budget']

    if args.mode.lower().startswith('r'):
        render_config = align_config['rendering']
        mode = 'rendering'
        num_workers = render_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            render_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('o'):
        optimization_config = align_config['optimization']
        mode = 'optimization'
        start_loc = optimization_config.get('slide_window', {}).get('start_loc', 'M')
        if start_loc.upper() == 'M':
            num_workers = min(2, optimization_config.get('slide_window', {}).get('num_workers', 2))
        else:
            num_workers = 1
        if num_workers > num_cpus:
            num_workers = num_cpus
            optimization_config.setdefault('slide_window', {})
            optimization_config['slide_window']['num_workers'] = num_workers
    elif args.mode.lower().startswith('ma'):
        mesh_config = align_config['meshing']
        match_config = align_config['matching']
        mode = 'matching'
        num_workers = match_config.get('matcher_config', {}).get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            match_config.setdefault('matcher_config', {})
            match_config['matcher_config']['num_workers'] = num_workers
            mesh_config['num_workers'] = min(num_workers, mesh_config.get('num_workers', num_workers))
    elif args.mode.lower().startswith('me'):
        mesh_config = align_config['meshing']
        mode = 'meshing'
        num_workers = mesh_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            mesh_config['num_workers'] = num_workers
    elif args.mode.lower().startswith('d'):
        min_mip = align_config.get('rendering', {}).get('mip_level', 0)
        mode = 'downsample'
        render_config = align_config.get('rendering', {})
        
        filename_config = {key:val for key, val in render_config.items() if key in ('pattern', 'one_based', 'tile_size')}
        if render_config.get('loader_config', {}).get('fillval', None) is not None:
            filename_config['fillval'] = render_config['loader_config']['fillval']
        filename_config.update(align_config['downsample'])
        #align_config = filename_config
        num_workers = align_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            align_config['num_workers'] = num_workers
        filename_config['min_mip'] = min_mip
    elif args.mode.lower().startswith('tensor'):
        tensor_config = align_config['tensorstore_rendering']
        mode = 'tensorstore_rendering'
        num_workers = tensor_config.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            tensor_config['num_workers'] = num_workers
    else:
        raise RuntimeError(f'{args.mode} not supported mode.')
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    align_dir = os.path.join(work_dir, 'align')
    mesh_dir = os.path.join(align_dir, 'mesh')
    match_dir = os.path.join(align_dir, 'matches')
    tform_dir = os.path.join(align_dir, 'tform')
    thumbnail_dir = os.path.join(work_dir, 'thumbnail_align')
    thumb_match_dir = os.path.join(thumbnail_dir, 'matches')
    render_dir = config.align_render_dir(args.work_dir)
    
    tensorstore_render_dir = config.tensorstore_render_dir(args.work_dir)
    ts_flag_dir = os.path.join(align_dir, 'ts_spec')
    thumbnail_configs = config.thumbnail_configs(args.work_dir)
    match_name_delimiter = thumbnail_configs.get('alignment', {}).get('match_name_delimiter', '__to__')
    align_config['mesh_dir'] = mesh_dir
    align_config['align_dir'] = align_dir
    align_config['match_dir'] = match_dir
    align_config['tform_dir'] = tform_dir
    align_config['thumbnail_dir'] = thumbnail_dir
    align_config['thumb_match_dir'] = thumb_match_dir
    align_config['render_dir'] = render_dir
    align_config['tensorstore_render_dir'] = tensorstore_render_dir
    align_config['ts_flag_dir'] = ts_flag_dir
    align_config['match_name_delimiter'] = match_name_delimiter
    align_config['work_dir'] = work_dir
    align_config['thumbnail_mip_level'] = thumbnail_configs.get('thumbnail_mip_level', 6)
    if args.mode.lower().startswith('r'):
        render_config['render_dir']=align_config['render_dir']
        return align_config, mode, render_config
    elif args.mode.lower().startswith('o'):
        return align_config, mode, optimization_config
    elif args.mode.lower().startswith('ma'):
        return align_config, mode, mesh_config, match_config
    elif args.mode.lower().startswith('me'):
        return align_config, mode, mesh_config
    elif args.mode.lower().startswith('d'):
        return align_config, mode, filename_config
    elif args.mode.lower().startswith('tensor'):
        return align_config, mode, tensor_config

def align_switchboard(mode, align_config, config_parts):
    if mode == 'meshing':
        os.makedirs(align_config['mesh_dir'], exist_ok=True)
        mesh_config = config_parts[2]
        generate_mesh_main(align_config,mesh_config)
    elif mode == 'matching':
        os.makedirs(align_config['mesh_dir'], exist_ok=True)
        mesh_config = config_parts[2]
        match_config=config_parts[3]
        start_matching =time.time()
        os.makedirs(align_config['match_dir'], exist_ok=True)
        generate_mesh_main(align_config,mesh_config)
        print("file lookup with glob")
        match_list = sorted(glob.glob(os.path.join(align_config['thumb_match_dir'], '*.h5')))
        assert len(match_list)>0, f"must find more than one match in {os.path.abspath(align_config['thumb_match_dir'])}"
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        align_config.setdefault('match_name_delimiter',  align_config['match_name_delimiter'])
        match_main(align_config, match_list)
        time_region.track_time('align_main.matching', time.time() - start_matching)
    elif mode == 'optimization':
        os.makedirs(align_config['tform_dir'], exist_ok=True)
        optimization_config = config_parts=[2]
        optimize_main(None, align_config, optimization_config)
    elif mode == 'rendering':
        render_config = config_parts[2]
        start_rendering =time.time() 
        if align_config.pop('offset_bbox', True):
            offset_name = os.path.join(align_config['tform_dir'], 'offset.txt')
            if not os.path.isfile(offset_name):
                time.sleep(0.1 * (1 + (args.start % args.step))) # avoid racing
                offset_bbox_main(align_config)
        os.makedirs(align_config['render_dir'], exist_ok=True)
        print("file lookup with glob")
        tform_list = sorted(glob.glob(os.path.join(align_config['tform_dir'], '*.h5')))
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        z_prefix = defaultdict(lambda: '')
        if align_config.pop('prefix_z_number', True):
            print("file lookup with glob")
            seclist = sorted(glob.glob(os.path.join(align_config['mesh_dir'], '*.h5')))
            section_order_file = os.path.join(align_config['work_dir'], 'section_order.txt')
            seclist, z_indx = common.rearrange_section_order(seclist, section_order_file)
            digit_num = math.ceil(math.log10(len(seclist)))
            z_prefix.update({os.path.basename(s): str(k).rjust(digit_num, '0')+'_'
                             for k, s in zip(z_indx, seclist)})
        render_main(tform_list, render_config, align_config['tform_dir'], z_prefix)
        time_region.track_time('align_main.rendering', time.time() - start_rendering)
    elif mode == 'downsample':
        filename_config = config_parts[2]
        min_mip = filename_config['min_mip']
        start_downsample = time.time()
        max_mip = align_config.pop('max_mip', 8)
        print("file lookup with glob")
        meta_list = sorted(glob.glob(os.path.join(align_config['render_dir'], 'mip'+str(min_mip), '**', 'metadata.txt'), recursive=True))
        meta_list = meta_list[indx]
        if args.reverse:
            meta_list = meta_list[::-1]
        generate_aligned_mipmaps(align_config['render_dir'], max_mip=max_mip, meta_list=meta_list, min_mip=min_mip, **filename_config)
        time_region.track_time('align_main.downsample', time.time() - start_downsample)
    elif mode == 'tensorstore_rendering':
        tensor_config = config_parts[2]
        start_tensorstore_rendering=time.time()
        logger_info = logging.initialize_main_logger(logger_name='tensorstore_render', mp=tensor_config['num_workers']>1)
        logger = logging.get_logger(logger_info[0])
        mip_level = tensor_config.pop('mip_level', 0)
        tensor_config.pop('outdir', None)
        driver = tensor_config.get('driver', 'neuroglancer_precomputed')
        if driver == 'zarr':
            tensorstore_render_dir = tensor_config['tensorstore_render_dir'] + '0/'
        elif driver == 'n5':
            tensorstore_render_dir = tensor_config['tensorstore_render_dir'] + 's0/'
        print("file lookup with glob")
        tform_list = sorted(glob.glob(os.path.join(tensor_config['tform_dir'], '*.h5')))
        section_order_file = os.path.join(tensor_config['work_dir'], 'section_order.txt')
        tform_list, z_indx = common.rearrange_section_order(tform_list, section_order_file)
        stitch_dir = os.path.join(tensor_config['work_dir'], 'stitch')
        loader_dir = os.path.join(stitch_dir, 'ts_specs')
        loader_list = [os.path.join(loader_dir, os.path.basename(s).replace('.h5', '.json')) for s in tform_list]
        resolution = config.DEFAULT_RESOLUTION * (2 ** mip_level)
        vol_renderer = VolumeRenderer(tform_list, loader_list, tensorstore_render_dir,
                                      z_indx = z_indx, resolution=resolution,
                                      flag_dir = tensor_config['ts_flag_dir'], **tensor_config)
        vol_renderer.render_volume(skip_indx=indx, logger=logger_info[0], **tensor_config)
        time_region.track_time('align_main.tensorstore_rendering', time.time() - start_tensorstore_rendering)
        logger.info('finished tensorstore_rendering')
        logging.terminate_logger(*logger_info)    

if __name__ == '__main__':
    args = parse_args()

    config_parts = setup_configs(args)
    align_config = config_parts[0]
    mode = config_parts[1]

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    indx = slice(stt_idx, stp_idx, step)

    align_switchboard(mode)
    time_region.log_summary()   