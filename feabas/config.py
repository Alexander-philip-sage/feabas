import math
import os
import yaml
from feabas import constant
from functools import lru_cache

if os.path.isfile(os.path.join(os.getcwd(), 'configs', 'general_configs.yaml')):
    _default_configuration_folder = os.path.join(os.getcwd(), 'configs')
elif os.path.isfile(os.path.join(os.path.dirname(os.getcwd()), 'configs', 'general_configs.yaml')):
    _default_configuration_folder = os.path.join(os.path.dirname(os.getcwd()), 'configs')
else:
    _default_configuration_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')

_default_log_dir = None

def print_default_log_dir():
    print("_default_log_dir",_default_log_dir)
def set_work_dir(work_dir):
    global _default_configuration_folder
    global _default_log_dir
    _default_configuration_folder = work_dir
    _default_log_dir = os.path.join(work_dir,"logs")  
    print("set_work_dir", _default_log_dir)  

@lru_cache(maxsize=1)
def general_settings(config_dir= _default_configuration_folder):
    config_file = os.path.join(config_dir, 'general_configs.yaml')
    if os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}
    if conf.get('cpu_budget', None) is None:
        import psutil
        conf['cpu_budget'] = psutil.cpu_count(logical=False)
    return conf


DEFAULT_RESOLUTION = general_settings().get('full_resolution', constant.DEFAULT_RESOLUTION)


@lru_cache(maxsize=1)
def get_work_dir():
    conf = general_settings()
    work_dir = conf.get('working_directory', './work_dir')
    global _default_log_dir
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    _default_log_dir = log_dir
    return work_dir


@lru_cache(maxsize=1)
def get_log_dir():
    if _default_log_dir:
        return _default_log_dir
    conf = general_settings()
    log_dir = conf.get('logging_directory', None)
    if log_dir is None:
        work_dir = conf.get('working_directory', './work_dir')
        log_dir = os.path.join(work_dir, 'logs')
    return log_dir


@lru_cache(maxsize=1)
def stitch_config_file(work_dir=None):
    if work_dir is None:
        work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'stitching_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join(_default_configuration_folder, 'default_stitching_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


@lru_cache(maxsize=1)
def stitch_configs(work_dir=None):
    if work_dir is None:
        with open(stitch_config_file(), 'r') as f:
            conf = yaml.safe_load(f)
    else:
        with open(stitch_config_file(work_dir), 'r') as f:
            conf = yaml.safe_load(f)        
    return conf


@lru_cache(maxsize=1)
def material_table_file(work_dir=None):
    if not work_dir:
        work_dir = get_work_dir()
    mt_file = os.path.join(work_dir, 'configs', 'material_table.json')
    if not os.path.isfile(mt_file):
        mt_file = os.path.join(_default_configuration_folder, 'default_material_table.json')
    return mt_file

@lru_cache(maxsize=1)
def align_config_file(root_dir=None):
    if root_dir:
        work_dir = root_dir
    else:
        work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'alignment_configs.yaml')
    if not os.path.isfile(config_file):
        config_file = os.path.join(_default_configuration_folder, 'default_alignment_configs.yaml')
        assert(os.path.isfile(config_file))
    return config_file


@lru_cache(maxsize=1)
def align_configs(root_dir=None):
    if root_dir:
        with open(align_config_file(root_dir), 'r') as f:
            conf = yaml.safe_load(f)
        section_thickness = general_settings(config_dir=os.path.join(root_dir, 'configs')).get('section_thickness', None)
    else:
        with open(align_config_file(), 'r') as f:
            conf = yaml.safe_load(f)
        section_thickness = general_settings().get('section_thickness', None)
    if (section_thickness is not None) and (conf.get('matching', {}).get('working_mip_level', None) is None):
        align_mip = max(0, math.floor(math.log2(section_thickness / DEFAULT_RESOLUTION)))
        conf.setdefault('matching', {})
        conf['matching'].setdefault('working_mip_level', align_mip)
    return conf


@lru_cache(maxsize=1)
def thumbnail_config_file(root_dir=None):
    if root_dir:
        work_dir = root_dir
    else:
        work_dir = get_work_dir()
    config_file = os.path.join(work_dir, 'configs', 'thumbnail_configs.yaml')
    #print("config file", config_file)
    if not os.path.isfile(config_file):
        print("couldn't find personal file at", config_file)
        config_file = os.path.join(_default_configuration_folder, 'default_thumbnail_configs.yaml')
        assert(os.path.isfile(config_file)), f"failed to find thumbnail config file at {config_file}"
    return config_file


@lru_cache(maxsize=1)
def thumbnail_configs(work_dir=None):
    with open(thumbnail_config_file(work_dir), 'r') as f:
        conf = yaml.safe_load(f)
    return conf


@lru_cache(maxsize=1)
def stitch_render_dir(work_dir=None):
    if work_dir is None:
        config_file = stitch_config_file()
    else:
        config_file = stitch_config_file(work_dir)
    with open(config_file, 'r') as f:        
        stitch_configs = yaml.safe_load(f)
    render_settings = stitch_configs.get('rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        if work_dir is None:
            work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'stitched_sections')
    return outdir


@lru_cache(maxsize=1)
def align_render_dir(work_dir=None):
    config_file = align_config_file(work_dir)
    with open(config_file, 'r') as f:        
        align_configs = yaml.safe_load(f)
    render_settings = align_configs.get('rendering', {})
    render_dir = render_settings.get('out_dir', None)
    if render_dir is None:
        if work_dir is None:
            work_dir = get_work_dir()
        render_dir = os.path.join(work_dir, 'aligned_stack')
    return render_dir


@lru_cache(maxsize=1)
def tensorstore_render_dir():
    config_file = align_config_file()
    with open(config_file, 'r') as f:        
        align_configs = yaml.safe_load(f)
    render_settings = align_configs.get('tensorstore_rendering', {})
    outdir = render_settings.get('out_dir', None)
    if outdir is None:
        work_dir = get_work_dir()
        outdir = os.path.join(work_dir, 'aligned_tensorstore')
    outdir = outdir.replace('\\', '/')
    if not outdir.endswith('/'):
        outdir = outdir + '/'
    kv_headers = ('gs://', 'http://', 'https://', 'file://', 'memory://')
    for kvh in kv_headers:
        if outdir.startswith(kvh):
            break
    else:
        outdir = 'file://' + outdir
    return outdir


def limit_numpy_thread(nthreads):
    nthread_str = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = nthread_str
    os.environ["OPENBLAS_NUM_THREADS"] = nthread_str
    os.environ["MKL_NUM_THREADS"] = nthread_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread_str
    os.environ["NUMEXPR_NUM_THREADS"] = nthread_str