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
import numpy as np
from typing import List

def generate_stitched_mipmaps(img_dir, max_mip,meta_list: List[str] =None, **kwargs):
    min_mip = kwargs.pop('min_mip', 0)
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    meta_dir = os.path.join(img_dir, 'mip'+str(min_mip), '**', 'metadata.txt')
    if meta_list is None:
        print("file lookup with glob")
        meta_list = sorted(glob.glob(meta_dir, recursive=True))
        assert len(meta_list)>0, f"did not find any metadata.txt files in {os.path.abspath(meta_dir)}"
        meta_list = meta_list[arg_indx]
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    if parallel_within_section or (num_workers == 1):
        for sname in secnames:
            mip_map_one_section(sname, img_dir, max_mip, num_workers=num_workers, **kwargs)
    else:
        target_func = partial(mip_map_one_section, img_dir=img_dir,
                                max_mip=max_mip, num_workers=1, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for sname in secnames:
                job = executor.submit(target_func, sname)
                jobs.append(job)
            for job in jobs:
                job.result()
    #logger.info('mipmapping generated.')


def generate_stitched_mipmaps_tensorstore(meta_dir, tgt_mips, meta_list: List[str]=None, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    parallel_within_section = kwargs.pop('parallel_within_section', True)
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    if meta_list is None:
        print("file lookup with glob")
        meta_regex = os.path.join(meta_dir,'*.json')
        meta_list = sorted(glob.glob(meta_regex))
        assert len(meta_list) > 0, f"did not find any json files in {os.path.abspath(meta_regex)}"
        meta_list = meta_list[arg_indx]
    if parallel_within_section or num_workers == 1:
        for metafile in meta_list:
            mipmap.generate_tensorstore_scales(metafile, tgt_mips, num_workers=num_workers, **kwargs)
    else:
        target_func = parallel_within_section(mipmap.generate_tensorstore_scales, mips=tgt_mips, **kwargs)
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for metafile in meta_list:
                job = executor.submit(target_func, metafile)
                jobs.append(job)
            for job in jobs:
                job.result()
    logger.info('mipmapping generated.')


def generate_thumbnails(src_dir, out_dir,meta_list=None, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    meta_regex = os.path.join(src_dir, '**', 'metadata.txt')
    assert len(meta_list)>0, f"couldn't find any metadata.txt files in {meta_regex}"
    if meta_list is None:
        print("file lookup with glob")
        meta_list = sorted(glob.glob(meta_regex, recursive=True))
        meta_list = meta_list[arg_indx]
    secnames = [os.path.basename(os.path.dirname(s)) for s in meta_list]
    target_func = partial(mipmap.create_thumbnail, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    updated = []
    if num_workers == 1:
        for sname in secnames:
            outname = os.path.join(out_dir, sname + '.png')
            if os.path.isfile(outname):
                continue
            updated.append(sname)
            sdir = os.path.join(src_dir, sname)
            img_out = target_func(sdir)
            common.imwrite(outname, img_out)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for sname in secnames:
                outname = os.path.join(out_dir, sname + '.png')
                if os.path.isfile(outname):
                    continue
                updated.append(sname)
                sdir = os.path.join(src_dir, sname)
                job = executor.submit(target_func, sdir, outname=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        #logger.info('thumbnails generated.')
    return updated


def generate_thumbnails_tensorstore(src_dir, out_dir, meta_list: List[str]=None, **kwargs):
    num_workers = kwargs.pop('num_workers', 1)
    logger_info = kwargs.pop('logger', None)
    logger = logging.get_logger(logger_info)
    if meta_list is None:
        print("file lookup with glob")
        meta_list = sorted(glob.glob(os.path.join(src_dir, '*.json')))
        meta_list = meta_list[arg_indx]
    target_func = partial(mipmap.create_thumbnail_tensorstore, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    updated = []
    if num_workers == 1:
        for meta_name in meta_list:
            sname = os.path.basename(meta_name).replace('.json', '')
            outname = os.path.join(out_dir, sname + '.png')
            if os.path.isfile(outname):
                continue
            updated.append(sname)
            img_out = target_func(meta_name)
            common.imwrite(outname, img_out)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for meta_name in meta_list:
                sname = os.path.basename(meta_name).replace('.json', '')
                outname = os.path.join(out_dir, sname + '.png')
                if os.path.isfile(outname):
                    continue
                updated.append(sname)
                job = executor.submit(target_func, meta_name, outname=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        logger.info('thumbnails generated.')
    return updated


def save_mask_for_one_sections(mesh_file, out_name, scale, **kwargs):
    from feabas.stitcher import MontageRenderer
    import numpy as np
    from feabas import common
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    rndr = MontageRenderer.from_h5(mesh_file)
    img = 255 - rndr.generate_roi_mask(scale, mask_erode=mask_erode)
    common.imwrite(out_name, img)
    if img_dir is not None:
        thumb_name = os.path.join(img_dir, os.path.basename(out_name))
        if os.path.isfile(thumb_name):
            thumb = common.imread(thumb_name)
            if (thumb.shape[0] != img.shape[0]) or (thumb.shape[1] != img.shape[1]):
                thumb_out_shape = (*img.shape, *thumb.shape[2:])
                thumb_out = np.full_like(thumb, fillval, shape=thumb_out_shape)
                mn_shp = np.minimum(thumb_out.shape[:2], thumb.shape[:2])
                thumb_out[:mn_shp[0], :mn_shp[1], ...] = thumb[:mn_shp[0], :mn_shp[1], ...]
                common.imwrite(thumb_name, thumb_out)


def generate_thumbnail_masks(out_dir,mesh_dir=None, mesh_list=None,seclist=None, **kwargs):
    num_workers = kwargs.get('num_workers', 1)
    scale = kwargs.get('scale')
    img_dir = kwargs.get('img_dir', None)
    fillval = kwargs.get('fillval', 0)
    mask_erode = kwargs.get('mask_erode', 0)
    logger_info = kwargs.get('logger', None)
    logger= logging.get_logger(logger_info)
    if (not mesh_list) and mesh_dir:
        arg_indx = kwargs.get("arg_indx")
        mesh_regex = os.path.abspath(os.path.join(mesh_dir, '*.h5'))
        print("file lookup with glob")
        mesh_list = sorted(glob.glob(mesh_regex))
        mesh_list = mesh_list[arg_indx]
    if not mesh_dir:
        raise ValueError("must pass either mesh list or mesh dir")
    assert len(mesh_list)>0, f"could not find an h5 files in mesh dir: mesh_regex {mesh_regex}"
    
    target_func = partial(save_mask_for_one_sections, scale=scale, img_dir=img_dir,
                          fillval=fillval, mask_erode=mask_erode)
    os.makedirs(out_dir, exist_ok=True)
    if num_workers == 1:
        for mname in mesh_list:
            sname = os.path.basename(mname).replace('.h5', '')
            outname = os.path.join(out_dir, sname + '.png')
            if seclist is None and os.path.isfile(outname):
                continue
            elif seclist is not None and sname not in seclist:
                continue
            target_func(mname, outname)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for mname in mesh_list:
                sname = os.path.basename(mname).replace('.h5', '')
                outname = os.path.join(out_dir, sname + '.png')
                if seclist is None and os.path.isfile(outname):
                    continue
                elif seclist is not None and sname not in seclist:
                    continue
                job = executor.submit(target_func, mname, out_name=outname)
                jobs.append(job)
            for job in jobs:
                job.result()
        #logger.info('thumbnail masks generated.')


def align_thumbnail_pairs(pairnames, image_dir, out_dir, **kwargs):
    import cv2
    import numpy as np
    from feabas import caching, thumbnail, common
    material_mask_dir = kwargs.pop('material_mask_dir', None)
    region_mask_dir = kwargs.pop('region_mask_dir', None)
    region_labels = kwargs.pop('region_labels', [0])
    match_name_delimiter = kwargs.pop('match_name_delimiter', '__to__')
    cache_size = kwargs.pop('cache_size', 3)
    feature_match_settings = kwargs.get('feature_matching', {})
    logger_info = kwargs.get('logger', None)
    logger = logging.get_logger(logger_info)
    prepared_cache = caching.CacheFIFO(maxlen=cache_size)
    for pname in pairnames:
        try:
            sname0_ext, sname1_ext = pname
            sname0 = os.path.splitext(sname0_ext)[0]
            sname1 = os.path.splitext(sname1_ext)[0]
            outname = os.path.join(out_dir, sname0 + match_name_delimiter + sname1 + '.h5')
            if os.path.isfile(outname):
                continue
            if sname0 in prepared_cache:
                minfo0 = prepared_cache[sname0]
            else:
                img0 = common.imread(os.path.join(image_dir, sname0_ext))
                if (region_mask_dir is not None) and os.path.isfile(os.path.join(region_mask_dir, sname0_ext)):
                    mask0 = common.imread(os.path.join(region_mask_dir, sname0_ext))
                elif (material_mask_dir is not None) and os.path.isfile(os.path.join(material_mask_dir, sname0_ext)):
                    mask_t = common.imread(os.path.join(material_mask_dir, sname0_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask0 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask0 = None
                minfo0 = thumbnail.prepare_image(img0, mask=mask0, **feature_match_settings)
                prepared_cache[sname0] = minfo0
            if sname1 in prepared_cache:
                minfo1 = prepared_cache[sname1]
            else:
                img1 = common.imread(os.path.join(image_dir, sname1_ext))
                if (region_mask_dir is not None) and os.path.isfile(os.path.join(region_mask_dir, sname1_ext)):
                    mask1 = common.imread(os.path.join(region_mask_dir, sname1_ext))
                elif (material_mask_dir is not None) and os.path.isfile(os.path.join(material_mask_dir, sname1_ext)):
                    mask_t = common.imread(os.path.join(material_mask_dir, sname1_ext))
                    mask_t = np.isin(mask_t, region_labels).astype(np.uint8)
                    _, mask1 = cv2.connectedComponents(mask_t, connectivity=4, ltype=cv2.CV_16U)
                else:
                    mask1 = None
                minfo1 = thumbnail.prepare_image(img1, mask=mask1, **feature_match_settings)
                prepared_cache[sname1] = minfo1
            thumbnail.align_two_thumbnails(minfo0, minfo1, outname, **kwargs)
        except Exception as err:
            logger.error(f'{pname}: error {err}')

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Align thumbnails")
    parser.add_argument("--work_dir", metavar="work_dir", type=str, default=None)    
    parser.add_argument("--mode", metavar="mode", type=str, default='downsample')
    parser.add_argument("--start", metavar="start", type=int, default=0)
    parser.add_argument("--step", metavar="step", type=int, default=1)
    parser.add_argument("--stop", metavar="stop", type=int, default=0)
    parser.add_argument("--reverse",  action='store_true')
    return parser.parse_args(args)

def setup_globals(args): 
    if not args.work_dir:
        root_dir = config.get_work_dir()
        generate_settings = config.general_settings()
        stitch_configs = config.stitch_configs()
    else:
        root_dir = args.work_dir
        config._default_configuration_folder = os.path.join(root_dir, "configs")
        generate_settings= config.general_settings(config._default_configuration_folder)
        stitch_configs = config.stitch_configs(root_dir)

    num_cpus = generate_settings['cpu_budget']

    thumbnail_configs = config.thumbnail_configs(root_dir)
    thumbnail_mip_lvl = thumbnail_configs.get('thumbnail_mip_level', 6)
    if args.mode.lower().startswith('d'):
        thumbnail_configs = thumbnail_configs['downsample']
        mode = 'downsample'
    elif args.mode.lower().startswith('a') or args.mode.lower().startswith('m'):
        thumbnail_configs = thumbnail_configs['alignment']
        mode = 'alignment'
    else:
        raise ValueError
    thumbnail_configs['work_dir'] = root_dir
    thumbnail_configs['thumbnail_mip_lvl'] = thumbnail_mip_lvl
    num_workers = thumbnail_configs.get('num_workers', 1)
    if num_workers > num_cpus:
        print("warning: num_workers has been reduced to the num_cpus found", num_cpus)
        num_workers = num_cpus
        thumbnail_configs['num_workers'] = num_workers
    nthreads = max(1, math.floor(num_cpus / num_workers))
    config.limit_numpy_thread(nthreads)

    thumbnail_dir = os.path.join(root_dir, 'thumbnail_align')
    stitch_tform_dir = os.path.join(root_dir, 'stitch', 'tform')
    thumbnail_img_dir = os.path.join(thumbnail_dir, 'thumbnails')
    mat_mask_dir = os.path.join(thumbnail_dir, 'material_masks')
    reg_mask_dir = os.path.join(thumbnail_dir, 'region_masks')
    manual_dir = os.path.join(thumbnail_dir, 'manual_matches')
    match_dir = os.path.join(thumbnail_dir, 'matches')
    feature_match_dir = os.path.join(thumbnail_dir, 'feature_matches')
    thumbnail_configs['thumbnail_img_dir'] = thumbnail_img_dir
    thumbnail_configs['stitch_tform_dir'] = stitch_tform_dir
    thumbnail_configs['mat_mask_dir'] = mat_mask_dir
    thumbnail_configs['feature_match_dir'] = feature_match_dir
    thumbnail_configs['match_dir'] = match_dir
    thumbnail_configs['manual_dir'] = manual_dir
    thumbnail_configs['reg_mask_dir'] = reg_mask_dir
    return (root_dir, generate_settings, num_cpus, thumbnail_configs, 
            thumbnail_mip_lvl, mode, num_workers, nthreads, thumbnail_dir, 
            stitch_tform_dir, thumbnail_img_dir, mat_mask_dir, reg_mask_dir, manual_dir, 
            match_dir, feature_match_dir)

def meta_list_to_mesh_list(meta_list, stitch_tform_dir):
    '''/eagle/BrainImagingML/apsage/feabas/work_dir4/stitched_sections/mip0/W02_Sec110_R1_montaged/metadata.txt'''
    sec_names = [x.split(os.path.sep)[-2] for x in meta_list]
    return [os.path.join(stitch_tform_dir, x+".h5") for x in sec_names]

def downsample_main(thumbnail_configs, work_dir=None,meta_list=None):
    start_downsample=time.time()
    logger_info = logging.initialize_main_logger(logger_name='stitch_mipmap', mp=thumbnail_configs.get('num_workers', 1)>1)
    thumbnail_configs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    align_mip = config.align_configs(work_dir)['matching']['working_mip_level']
    stitch_conf = config.stitch_configs(work_dir)['rendering']
    driver = stitch_conf.get('driver', 'image')
    stitch_tform_dir=thumbnail_configs['stitch_tform_dir']
    thumbnail_mip_lvl = thumbnail_configs['thumbnail_mip_lvl']
    thumbnail_img_dir=thumbnail_configs['thumbnail_img_dir']
    mat_mask_dir=thumbnail_configs['mat_mask_dir']
    feature_match_dir=thumbnail_configs['feature_match_dir']
    match_dir =thumbnail_configs['match_dir']    
    if 'thumbnail_img_dir' in thumbnail_configs.keys():
        thumbnail_img_dir = thumbnail_configs['thumbnail_img_dir']
    if driver == 'image':
        max_mip = thumbnail_configs.pop('max_mip', max(0, thumbnail_mip_lvl-1))
        max_mip = max(align_mip, max_mip)
        src_dir0 = config.stitch_render_dir(work_dir)
        #print("src_dir0", src_dir0)
        pattern = stitch_conf['filename_settings']['pattern']
        one_based = stitch_conf['filename_settings']['one_based']
        fillval = stitch_conf['loader_settings'].get('fillval', 0)
        thumbnail_configs.setdefault('pattern', pattern)
        thumbnail_configs.setdefault('one_based', one_based)
        thumbnail_configs.setdefault('fillval', fillval)
        generate_stitched_mipmaps(src_dir0, max_mip,meta_list=meta_list, **thumbnail_configs)
        if thumbnail_configs.get('thumbnail_highpass', True):
            src_mip = max(0, thumbnail_mip_lvl-2)
            highpass_inter_mip_lvl = thumbnail_configs.get('highpass_inter_mip_lvl', src_mip)
            assert highpass_inter_mip_lvl < thumbnail_mip_lvl
            src_dir = os.path.join(src_dir0, 'mip'+str(highpass_inter_mip_lvl))
            downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
            if downsample >= 4:
                highpass = True
            else:
                highpass = False
        else:
            src_mip = max(0, thumbnail_mip_lvl-1)
            src_dir = os.path.join(src_dir0, 'mip'+str(src_mip))
            downsample = 2 ** (thumbnail_mip_lvl - src_mip)
            highpass = False
        thumbnail_configs.setdefault('downsample', downsample)
        thumbnail_configs.setdefault('highpass', highpass)
        slist = generate_thumbnails(src_dir, thumbnail_img_dir,meta_list=meta_list, **thumbnail_configs)
    elif driver =='neuroglancer_precomputed':
        stitch_dir = os.path.join(thumbnail_configs['work_dir'], 'stitch')
        meta_dir = os.path.join(stitch_dir, 'ts_specs')
        tgt_mips = [align_mip]
        if thumbnail_configs.get('thumbnail_highpass', True):
            highpass_inter_mip_lvl = thumbnail_configs.get('highpass_inter_mip_lvl', max(0,thumbnail_mip_lvl-2))
            assert highpass_inter_mip_lvl < thumbnail_mip_lvl
            downsample = 2 ** (thumbnail_mip_lvl - highpass_inter_mip_lvl)
            if downsample >= 4:
                highpass = True
                thumbnail_configs.setdefault('highpass_inter_mip_lvl', highpass_inter_mip_lvl)
                tgt_mips.append(highpass_inter_mip_lvl)
            else:
                highpass = False
                tgt_mips.append(thumbnail_mip_lvl)
                downsample = 1
        else:
            highpass = False
            tgt_mips.append(thumbnail_mip_lvl)
            downsample = 1
        generate_stitched_mipmaps_tensorstore(meta_dir, tgt_mips,meta_list=meta_list, **thumbnail_configs)
        thumbnail_configs.setdefault('highpass', highpass)
        thumbnail_configs.setdefault('mip', thumbnail_mip_lvl)
        slist = generate_thumbnails_tensorstore(meta_dir, thumbnail_img_dir,meta_list=meta_list, **thumbnail_configs)
    else:
        raise NotImplementedError("saving with other file types is not tested")
    mask_scale = 1 / (2 ** thumbnail_mip_lvl)
    if meta_list is not None:
        mesh_list = meta_list_to_mesh_list(meta_list, stitch_tform_dir)
    else:
        mesh_list=None
    generate_thumbnail_masks(mat_mask_dir,mesh_dir=stitch_tform_dir, mesh_list=mesh_list, seclist=slist, scale=mask_scale,
                                img_dir=thumbnail_img_dir, **thumbnail_configs)
    generate_thumbnail_masks( mat_mask_dir, mesh_dir=stitch_tform_dir, mesh_list=mesh_list, seclist=None, scale=mask_scale,
                                img_dir=thumbnail_img_dir, **thumbnail_configs)
    time_region.track_time('thumbnail_main.downsample', time.time() - start_downsample)
    #logger.info('finished thumbnail downsample.')
    logging.terminate_logger(*logger_info)
def setup_bnames(img_dir: str,work_dir: str, imglist: List[str] =None):
    if not imglist:
        img_regex = os.path.abspath(os.path.join(img_dir, '*.png'))
        imglist = sorted(glob.glob(img_regex))
        assert len(imglist)>0, f"couldn't find any png files in {img_regex}"
    print("img_dir",img_dir )
    print("work_dir", work_dir)
    section_order_file = os.path.join(work_dir, 'section_order.txt')
    imglist = common.rearrange_section_order(imglist, section_order_file)[0]
    bname_list = [os.path.basename(s) for s in imglist]
    return imglist, bname_list
def setup_pairnames(bname_list: List[str], compare_distance: int):
    pairnames = []
    for stp in range(1, compare_distance+1):
        for k in range(len(bname_list)-stp):
            pairnames.append((bname_list[k], bname_list[k+stp]))
    pairnames.sort()
    return pairnames
def setup_pair_names(img_dir: str,work_dir: str,  compare_distance: int, imglist: List[str] =None):
    imglist, bname_list = setup_bnames(img_dir, work_dir, imglist=imglist)
    print("bname_list", bname_list)
    pairnames = setup_pairnames(bname_list, compare_distance)
    return imglist, bname_list, pairnames

def align_main(thumbnail_configs,pairnames=None, num_workers:int =None):
    start_alignment = time.time()
    work_dir = thumbnail_configs['work_dir']
    match_dir = thumbnail_configs['match_dir']
    manual_dir = thumbnail_configs['manual_dir']
    mat_mask_dir = thumbnail_configs['mat_mask_dir']
    thumbnail_mip_lvl=thumbnail_configs['thumbnail_mip_lvl']
    thumbnail_img_dir = thumbnail_configs['thumbnail_img_dir']
    reg_mask_dir = thumbnail_configs['reg_mask_dir']
    os.makedirs(match_dir, exist_ok=True)
    os.makedirs(manual_dir, exist_ok=True)
    compare_distance = thumbnail_configs.pop('compare_distance', 1)
    if pairnames is None:
        imglist, bname_list, pairnames = setup_pair_names(thumbnail_img_dir,work_dir,  compare_distance)
        pairnames = pairnames[arg_indx]
    logger_info = logging.initialize_main_logger(logger_name='thumbnail_align', mp=num_workers>1)
    thumbnail_configs['logger'] = logger_info[0]
    logger= logging.get_logger(logger_info[0])
    resolution = config.DEFAULT_RESOLUTION * (2 ** thumbnail_mip_lvl)
    thumbnail_configs.setdefault('resolution', resolution)
    #thumbnail_configs.setdefault('feature_match_dir', feature_match_dir)
    region_labels = []
    material_table_file = config.material_table_file(thumbnail_configs['work_dir'])
    material_table = material.MaterialTable.from_json(material_table_file, stream=False)
    for _, mat in material_table:
        if mat.enable_mesh and (mat._stiffness_multiplier > 0.1) and (mat.mask_label is not None):
            region_labels.append(mat.mask_label)
    thumbnail_configs.setdefault('region_labels', region_labels)
    target_func = partial(align_thumbnail_pairs, image_dir=thumbnail_img_dir, out_dir=match_dir,
                            material_mask_dir=mat_mask_dir, region_mask_dir=reg_mask_dir,
                            **thumbnail_configs)
    if (num_workers == 1) or (len(pairnames) <= 1):
        target_func(pairnames)
    else:
        num_workers = min(num_workers, len(pairnames))
        match_per_job = thumbnail_configs.pop('match_per_job', 15)
        Njobs = max(num_workers, len(pairnames) // match_per_job)
        indx_j = np.linspace(0, len(pairnames), num=Njobs+1, endpoint=True)
        indx_j = np.unique(np.round(indx_j).astype(np.int32))
        jobs = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('fork')) as executor:
            for idx0, idx1 in zip(indx_j[:-1], indx_j[1:]):
                prnm = pairnames[idx0:idx1]
                job = executor.submit(target_func, pairnames=prnm)
                jobs.append(job)
            for job in jobs:
                job.result()
    time_region.track_time('thumbnail_main.alignment', time.time() - start_alignment)
    #logger.info('finished thumbnail alignment.')
    logging.terminate_logger(*logger_info)

if __name__ == '__main__':
    args = parse_args()

    root_dir, generate_settings, num_cpus, thumbnail_configs, thumbnail_mip_lvl, mode, num_workers, nthreads, thumbnail_dir, stitch_tform_dir, img_dir, mat_mask_dir, reg_mask_dir, manual_dir, match_dir, feature_match_dir = setup_globals(args)

    stt_idx, stp_idx, step = args.start, args.stop, args.step
    if stp_idx == 0:
        stp_idx = None
    if args.reverse:
        if stt_idx == 0:
            stt_idx = None
        arg_indx = slice(stp_idx, stt_idx, -step)
    else:
        arg_indx = slice(stt_idx, stp_idx, step)
    thumbnail_configs['arg_indx']=arg_indx
    if mode == 'downsample':
        downsample_main(thumbnail_configs)
    elif mode == 'alignment':
        assert num_workers, "num_workers must have a value"
        #print("num_workers", num_workers)
        align_main(thumbnail_configs,num_workers=num_workers)
    time_region.log_summary()

