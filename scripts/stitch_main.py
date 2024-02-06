import argparse
import glob
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import get_context
import math
from functools import partial
import os
import time, datetime
import tensorstore as ts
import feabas
from feabas import config, logging, dal
from feabas.time_region import time_region
from feabas.time_region import timer_func
def match_one_section(coordname, outname, **kwargs):
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    stitcher = Stitcher.from_coordinate_file(coordname)
    if os.path.isfile(outname + '_err'):
        logger.info(f'loading previous results for {os.path.basename(coordname)}')
        stitcher.load_matches_from_h5(outname + '_err', check_order=True)
    _, err = stitcher.dispatch_matchers(verbose=False, **kwargs)
    if err:
        outname = outname + '_err'
    stitcher.save_to_h5(outname, save_matches=True, save_meshes=False)
    return 1

@timer_func
def stitch_match_main(coord_list, out_dir, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    logger_info = logging.initialize_main_logger(logger_name='stitch_matching', mp=num_workers>1)
    kwargs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    for coordname in coord_list:
        t0 = time.time()
        fname = os.path.basename(coordname).replace('.txt', '')
        outname = os.path.join(out_dir, fname + '.h5')
        if os.path.isfile(outname):
            continue
        logger.info(f'starting matching for {fname}')
        flag = match_one_section(coordname, outname, **kwargs)
        if flag == 1:
            logger.info(f'ending for {fname}: {(time.time()-t0)/60} min')
    logger.info('finished.')
    logging.terminate_logger(*logger_info)


def optimize_one_section(matchname, outname, **kwargs):
    from feabas.stitcher import Stitcher
    import numpy as np
    use_group = kwargs.get('use_group', True)
    msem = kwargs.get('msem', False)
    mesh_settings = kwargs.get('mesh_settings', {})
    translation_settings = kwargs.get('translation', {})
    group_elastic_settings = kwargs.get('group_elastic', {})
    elastic_settings = kwargs.get('final_elastic', {})
    disconnected_settings = kwargs.get('disconnected_assemble', {})
    normalize_setting = kwargs.get('normalize', {})
    minweight = kwargs.get('minweight', None)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    group_elastic_settings.setdefault('continue_on_flip', True)
    elastic_settings.setdefault('continue_on_flip', True)
    bname = os.path.basename(matchname).replace('.h5', '')
    t0 = time.time()
    stitcher = Stitcher.from_h5(matchname, load_matches=True, load_meshes=False)
    if minweight is not None:
        rejected = stitcher.filter_match_by_weight(minweight)
        if rejected > 0:
            logger.debug(f'{bname}: filtered out {rejected} low-conf matches')
    if use_group:
        if msem:
            groupings = [int(s.split('/')[0]) for s in stitcher.imgrelpaths]
        else:
            groupings = np.zeros(stitcher.num_tiles, dtype=np.int32)
    else:
        groupings = None 
    stitcher.set_groupings(groupings)
    mesh_settings = mesh_settings.copy()
    mesh_sizes = mesh_settings.pop('mesh_sizes', [75, 150, 300])
    stitcher.initialize_meshes(mesh_sizes, **mesh_settings)
    discrd =stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
    dis = stitcher.match_residues()
    logger.info(f'{bname}: residue after translation {np.nanmean(dis)} | discarded {discrd}')
    if use_group:
        stitcher.optimize_group_intersection(target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        stitcher.optimize_translation(target_gear=feabas.MESH_GEAR_FIXED, **translation_settings)
        cost = stitcher.optimize_elastic(use_groupings=True, target_gear=feabas.MESH_GEAR_FIXED, **group_elastic_settings)
        dis = stitcher.match_residues()
        logger.info(f'{bname}: residue after grouped relaxation {np.nanmean(dis)} | cost {cost}')
    cost = stitcher.optimize_elastic(target_gear=feabas.MESH_GEAR_MOVING, **elastic_settings)
    rot, _ = stitcher.normalize_coordinates(**normalize_setting)
    N_conn = stitcher.connect_isolated_subsystem(**disconnected_settings)
    if N_conn > 1:
        rot_1, _ = stitcher.normalize_coordinates(**normalize_setting)
        rot = max(rot, rot_1)
    if cost[0] is None or cost[1] is None or cost[0] < cost[1]:
        stitcher.save_to_h5(outname.replace('.h5', '.h5_err'), save_matches=False, save_meshes=True)
        logger.error(f'{bname}: failed to converge.')
    else:
        stitcher.save_to_h5(outname, save_matches=False, save_meshes=True)
    lbl_conn = stitcher.connected_subsystem
    _, cnt = np.unique(lbl_conn, return_counts=True)
    ncomp = cnt.size
    ncomp1 = np.sum(cnt>1)
    dis = stitcher.match_residues()
    finish_str = f'{bname}: {cost} finished {time.time() - t0} sec | residue: {np.nanmean(dis)} | rotation: {round(rot)}'
    if abs(rot) > 1.5:
        logger.warning(f'{bname}: rotation detected in final transform, potential mesh relaxation issues.')
    if ncomp > 1:
        finish_str = finish_str + f' | {ncomp1}/{ncomp} components'
        logger.warning(f'\t{bname}: {ncomp} disconnected groups found, among which {ncomp1} have more than one tiles.')
    logger.info(finish_str)

@timer_func
def stitch_optmization_main(match_list, out_dir, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = logging.initialize_main_logger(logger_name='stitch_optmization', mp=num_workers>1)
    kwargs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    target_func = partial(optimize_one_section, **kwargs)
    if num_workers == 1:
        for matchname in match_list:
            outname = os.path.join(out_dir, os.path.basename(matchname))
            if os.path.isfile(outname):
                continue
            target_func(matchname, outname)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for matchname in match_list:
                outname = os.path.join(out_dir, os.path.basename(matchname))
                if os.path.isfile(outname):
                    continue
                job = executor.submit(target_func, matchname, outname)
                jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('finished.')
    logging.terminate_logger(*logger_info)


def render_one_section(tform_name, out_prefix, meta_name=None, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    tile_size = kwargs.pop('tile_size', [4096, 4096])
    scale = kwargs.pop('scale', 1.0)
    resolution = kwargs.pop('resolution', None)
    loader_settings = kwargs.get('loader_settings', {})
    render_settings = kwargs.get('render_settings', {}).copy()
    driver = kwargs.get('driver', 'image')
    use_tensorstore = driver != 'image'
    if loader_settings.get('cache_size', None) is not None:
        loader_settings = loader_settings.copy()
        loader_settings['cache_size'] = loader_settings['cache_size'] // num_workers
    if meta_name is not None and os.path.isfile(meta_name):
        return None
    renderer = MontageRenderer.from_h5(tform_name, loader_settings=loader_settings)
    if resolution is not None:
        scale = renderer.resolution / resolution
    else:
        resolution = renderer.resolution / scale
    render_settings['scale'] = scale
    out_prefix = out_prefix.replace('\\', '/')
    print(f"out_prefix {out_prefix}")
    render_series = renderer.plan_render_series(tile_size, prefix=out_prefix,
        scale=scale, **kwargs)
    if use_tensorstore:
        # delete existing
        out_spec = render_series[1].copy()
        out_spec.update({'open': False, 'create': True, 'delete_existing': True})
        store = ts.open(out_spec).result()
    if num_workers == 1:
        bboxes, filenames, _ = render_series
        metadata = renderer.render_series_to_file(bboxes, filenames, **render_settings)
    else:
        bboxes_list, filenames_list, hits_list = renderer.divide_render_jobs(render_series,
            num_workers=num_workers, max_tile_per_job=20)
        if use_tensorstore:
            metadata = []
        else:
            metadata = {}
        jobs = []
        target_func = partial(MontageRenderer.subprocess_render_montages, **render_settings)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('spawn')) as executor:
            for bboxes, filenames, hits in zip(bboxes_list, filenames_list, hits_list):
                init_args = renderer.init_args(selected=hits)
                job = executor.submit(target_func, init_args, bboxes, filenames)
                jobs.append(job)
            for job in as_completed(jobs):
                if use_tensorstore:
                    metadata.extend(job.result())
                else:
                    metadata.update(job.result())
    if (meta_name is not None) and (len(metadata) > 0):
        if use_tensorstore:
            meta_name = meta_name.replace('\\', '/')
            kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://')
            for kvh in kv_headers:
                if meta_name.startswith(kvh):
                    break
            else:
                meta_name = 'file://' + meta_name
            meta_ts = ts.open({"driver": "json", "kvstore": meta_name}).result()
            meta_ts.write({0: store.spec(minimal_spec=True).to_json()}).result()
        else:
            fnames = sorted(list(metadata.keys()))
            bboxes = []
            for fname in fnames:
                bboxes.append(metadata[fname])
            out_loader = dal.StaticImageLoader(fnames, bboxes=bboxes, resolution=resolution)
            out_loader.to_coordinate_file(meta_name)
    return len(metadata)

@timer_func
def stitch_render_main(tform_list, out_dir, **kwargs):
    logger_info = logging.initialize_main_logger(logger_name='stitch_rendering', mp=False)
    logger = logger_info[0]
    driver = kwargs.get('driver', 'image')
    use_tensorstore = driver != 'image'
    if use_tensorstore:
        meta_dir = kwargs['meta_dir']
        os.makedirs(meta_dir, exist_ok=True)
    for tname in tform_list:
        t0 = time.time()
        sec_name = os.path.basename(tname).replace('.h5', '')
        try:
            sec_outdir = os.path.join(out_dir, sec_name)
            if use_tensorstore:
                meta_name = os.path.join(meta_dir, sec_name+'.json')
            else:
                meta_name = os.path.join(sec_outdir, 'metadata.txt')
            if os.path.isfile(meta_name):
                continue
            logger.info(f'{sec_name}: start')
            if use_tensorstore:
                out_prefix = sec_outdir
            else:
                if  sec_outdir.startswith('file://'):
                    sec_outdir = sec_outdir.replace('file://', '')
                os.makedirs(sec_outdir, exist_ok=True)
                out_prefix = os.path.join(sec_outdir, sec_name)
            num_rendered = render_one_section(tname, out_prefix, meta_name=meta_name, **kwargs)
            logger.info(f'{sec_name}: {num_rendered} tiles | {(time.time()-t0)/60} min')
        except Exception as err:
            logger.error(f'{sec_name}: {err}')
    logger.info('finished.')
    logging.terminate_logger(*logger_info)

def look_for_duplicate_sections(coord_list):
    basenames = [os.path.basename(x) for x in coord_list]
    section_names = set()
    for bn in basenames:
        parts = bn.split('_')
        assert len(parts)>=3, "expected section tilespec file to be of form 'W03_Sec106_R2_montaged.txt'"
        sn = ''.join(parts[:2])
        if sn in section_names:
            raise Exception(f"ERROR: two sections with the same sec name {sn} {bn}\nThis is likely representative of a duplicate section")

def stitch_switchboard(mode):
    if mode in ['rendering', 'render']:
        stitch_configs_render = stitch_configs['rendering']
        stitch_configs_render.pop('out_dir', '')
        image_outdir = config.stitch_render_dir()
        driver = stitch_configs_render.get('driver', 'image')
        if driver == 'image':
            image_outdir = os.path.join(image_outdir, 'mip0')
        tform_regex = os.path.abspath(os.path.join(mesh_dir, '*.h5'))
        tform_list = sorted(glob.glob(tform_regex))
        assert len(tform_list)>0, f"tform list empty, didn't find any h5 files in {tform_regex}"
        tform_list = tform_list[indx]
        if args.reverse:
            tform_list = tform_list[::-1]
        stitch_configs_render.setdefault('meta_dir', render_meta_dir)
        print(f"image_outdir {image_outdir}")
        stitch_render_main(tform_list, image_outdir, **stitch_configs_render)

    elif mode in ['optimization', 'optimize']:
        print("starting optimize")
        stitch_configs_opt = stitch_configs['optimization']
        match_regex = os.path.abspath(os.path.join(match_dir, '*.h5'))
        match_list = sorted(glob.glob(match_regex))
        assert len(match_list) > 0, f"match list couldn't find any h5 files {match_regex}"
        match_list = match_list[indx]
        if args.reverse:
            match_list = match_list[::-1]
        os.makedirs(mesh_dir, exist_ok=True)
        stitch_optmization_main(match_list, mesh_dir, **stitch_configs_opt)

    elif mode in ['matching', 'match']:
        stitch_configs_match = stitch_configs['matching']
        coord_regex = os.path.abspath(os.path.join(coord_dir, '*.txt'))
        coord_list = sorted(glob.glob(coord_regex))
        assert len(coord_list) > 0, f"didn't find any txt coord files {coord_regex}"
        look_for_duplicate_sections(coord_list)
        coord_list = coord_list[indx]
        if args.reverse:
            coord_list = coord_list[::-1]
        os.makedirs(match_dir, exist_ok=True)
        stitch_match_main(coord_list, match_dir, **stitch_configs_match)    
        

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run stitching")
    parser.add_argument("--mode", metavar="mode", type=str, default='matching')
    parser.add_argument("--work_dir", metavar="work_dir", type=str, default=None)
    #parser.add_argument("--section_name", metavar="section_name", type=str, default=None)
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0, 
                        help="stop index is exclusive. stop of one batch should be the same as the start of the next")
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    if not args.work_dir:
        root_dir = config.get_work_dir()
    else:
        root_dir = args.work_dir
        config.set_work_dir(args.work_dir)
        logging.set_log_configs()
    general_settings= config.general_settings()
    stitch_configs = config.stitch_configs()
    print('os.getcwd()',os.getcwd())
    num_cpus = general_settings['cpu_budget']
    print("root_dir", root_dir)
    #print("general_settings", general_settings)
    #print("stitch_configs", stitch_configs)
    print(datetime.datetime.now(), "start mode stitch", args.mode)
    if args.mode.lower().startswith('r'):
        mode = 'rendering'
    elif args.mode.lower().startswith('o'):
        mode = 'optimization'
    elif args.mode.lower().startswith('m'):
        mode = 'matching'
    else:
        mode='all'
    if mode=='all':
        num_workers = stitch_configs.get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            stitch_configs['num_workers'] = num_workers
    else:
        num_workers = stitch_configs[mode].get('num_workers', 1)
        if num_workers > num_cpus:
            num_workers = num_cpus
            stitch_configs[mode]['num_workers'] = num_workers
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    from feabas.stitcher import Stitcher, MontageRenderer
    import numpy as np

    stitch_dir = os.path.join(root_dir, 'stitch')
    coord_dir = os.path.join(stitch_dir, 'stitch_coord')
    match_dir = os.path.join(stitch_dir, 'match_h5')
    mesh_dir = os.path.join(stitch_dir, 'tform')
    render_meta_dir = os.path.join(stitch_dir, 'ts_specs')
    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    indx = slice(stt_idx, stp_idx, step)
    if mode =='all':
        stitch_switchboard('matching')
        stitch_switchboard('optimization')
        stitch_switchboard('rendering')
    else:
        stitch_switchboard(mode)
    print(datetime.datetime.now(), "finish mode stitch", args.mode)
    time_region.log_summary()