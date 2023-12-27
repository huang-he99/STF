#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/STIL/model6/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_200~ddim_50~pred_noise.py

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/STIL/model6/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_200~ddim_50~pred_x0.py


#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/STIL/model6/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_noise.py

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/STIL/model6/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py


#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_BN_ReLU/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_BN_SiLU/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_GN_ReLU/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_GN_SiLU/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py


#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6/LGC/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_BN_SiLU/LGC/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_BN_ReLU/LGC/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_GN_ReLU/LGC/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py  ###

#   python tools/train/train_bi_stfdiff.py --congfig_path config/bi-stfdiff/syy_setting-9/model6_GN_SiLU/LGC/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import importlib
import warnings

from src.trainer.bi_stfdiff_trainer import Trainer
from src.utils import fix_random_seed

warnings.filterwarnings("ignore")

fix_random_seed(42)

parser = argparse.ArgumentParser(description="data generation")
parser.add_argument("--congfig_path", type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace("/", ".").replace(".py", "")
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace("config", "results").replace(".py", "")


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
