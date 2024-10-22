# python tools/train/train_stfnet.py --congfig_path config/stfnet/syy_setting-8/SGD_5e-4.py
# python tools/train/train_stfnet.py --congfig_path config/stfnet/syy_setting-8/SGD_5e-4~StepLR_40.py
# python tools/train/train_stfnet.py --congfig_path config/stfnet/syy_setting-8/SGD_1e-2~StepLR_40.py
# python tools/train/train_stfnet.py --congfig_path config/stfnet/syy_setting-8/SGD_1e-3~StepLR_40.py
# python tools/train/train_stfnet.py --congfig_path config/stfnet/syy_setting-8/SGD_1e-1~StepLR_40.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import importlib
import warnings

from src.trainer.stfnet_trainer import Trainer
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
work_dir = congfig_path.replace('config', 'results').replace('.py', '')


exp = Trainer(
    congfig_path=congfig_path,
    train_root_dir_path=work_dir,
    train_dataloader=config.train_dataloader,
    val_dataloader=config.val_dataloader,
    model=config.model,
    optimizer=config.optimizer,
    scheduler=config.scheduler,
    metric_list=config.metric_list,
)
exp.train()
