# python tools/inference/test_starfm.py --congfig_path config/starfm/syy_setting~9/CIA/patch_size_120~patch_stride_50~window_size_51~num_classes_20.py


# python tools/inference/test_starfm.py --congfig_path config/starfm/syy_setting~9/CIA/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20.py


# python tools/inference/test_starfm.py --congfig_path config/starfm/syy_setting~9/LGC/patch_size_120~patch_stride_50~window_size_51~num_classes_20.py


# python tools/inference/test_starfm.py --congfig_path config/starfm/syy_setting~9/LGC/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20.py


# python tools/inference/test_starfm.py --congfig_path config/starfm/STIL/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20.py
# python tools/inference/test_starfm.py --congfig_path config/starfm/STIL/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_4.py

# python tools/inference/test_starfm.py --congfig_path config/starfm/syy_setting-8.py


import argparse
import importlib
import os
import warnings

os.environ['OMP_NUM_THREADS'] = '1'

from src.inferencer.starfm_inferencer import Inferencer
from src.utils import fix_random_seed

warnings.filterwarnings("ignore")

fix_random_seed(42)

parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('--congfig_path', type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace('/', '.').replace('.py', '')
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace('config', 'results').replace('.py', '/')
img_mode = config.test_dataloader.dataset.data_root.name
work_dir = work_dir + img_mode

exp = Inferencer(
    congfig_path=congfig_path,
    inference_root_dir_path=work_dir,
    test_dataloader=config.test_dataloader,
    model=config.model,
    metric_list=config.metric_list,
    patch_info_dict=config.patch_info_dict,
)
exp.inference()
