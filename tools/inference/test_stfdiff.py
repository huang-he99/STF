# python tools/inference/test_opgan.py --congfig_path config/opgan/syy_setting-9/CIA/inference_full~epoch_349~RMSProp_1e-4~StepLR_1e-1.py


#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_200~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_200~ddim_50~pred_noise.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_BN_ReLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_BN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_GN_ReLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_GN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/setting-10/model6_GN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py
#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_BN_ReLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py
#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_BN_SiLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py
#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_GN_ReLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py
#  python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/syy_setting-9/model6_GN_SiLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

# python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

# python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6_BN_ReLU/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

# python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6_BN_SiLU/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

# python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6_GN_ReLU/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py

# python tools/inference/test_stfdiff.py --congfig_path config/stfdiff/STIL/model6_GN_SiLU/Inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0.py


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['OMP_NUM_THREADS'] = '1'

from src.inferencer.stfdiff_inferencer import Inferencer
import argparse
import importlib
import warnings
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
work_dir = congfig_path.replace("config", "results").replace(".py", "/")
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
