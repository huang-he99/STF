import os
from src.trainer.starfm_trainer import Experiment
import argparse
import importlib

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# python train_2.py --congfig_path config/starfm/starfm~CIA_syy_setting-8.py


parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('--congfig_path', type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace('/', '.').replace('.py', '')
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace('config', 'results').replace('.py', '')


exp = Experiment(
    experiment_root_dir=work_dir,
    test_dataloader=config.test_dataloader,
    model=config.model,
    metric_list=config.metric_list,
    patch_generator=config.patch_generator,
)
exp.test()
