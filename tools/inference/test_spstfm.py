# python tools/inference/test_spstfm.py --congfig_path config/spstfm/syy_setting_8/sample_num-2000~atom_num-512~sparsity-10/reconstruction.py
import argparse
import importlib
import os
import warnings

# os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.inferencer.spstfm_inferencer import Inferencer
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
img_mode = config.test_dataloader.dataset.datasets[0].data_root.name
work_dir = work_dir + img_mode

exp = Inferencer(
    congfig_path=congfig_path,
    inference_root_dir_path=work_dir,
    test_dataloader=config.test_dataloader,
    model=config.model,
    metric_list=config.metric_list,
)
exp.inference()
