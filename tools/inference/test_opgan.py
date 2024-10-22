# python tools/inference/test_opgan.py --congfig_path config/opgan/syy_setting-9/CIA/inference~RMSProp.py

# python tools/inference/test_opgan.py --congfig_path config/opgan/setting-10/CIA/inference~RMSProp.py

# python tools/inference/test_opgan.py --congfig_path config/opgan/syy_setting-9/LGC/inference~RMSProp.py

# python tools/inference/test_opgan.py --congfig_path config/opgan/STIL/inference~RMSProp.py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import importlib
import warnings

from src.inferencer.opgan_inferencer import Inferencer
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
    checkpoint_path=config.checkpoint_path,
    metric_list=config.metric_list,
)
exp.inference()
