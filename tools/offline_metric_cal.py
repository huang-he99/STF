from pathlib import Path
from src.metrics import *
from src.logger import FusionLogger
import tifffile as tiff
import numpy as np
import torch

gt_dir_path = Path(
    'data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/test/full/Landsat_02'
)
pred_dir_path = Path('results/stfdcnn/syy_setting-8/syy_results/LGC')

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(data_range=1.0),
]


txt_logger = FusionLogger(
    logger_name='offline_metric_cal',
    log_level='INFO',
)

gt_img_path_list = sorted(list(gt_dir_path.glob('*.tif')))
pred_img_path_list = sorted(list(pred_dir_path.glob('*.tif')))

assert len(gt_img_path_list) == len(pred_img_path_list)
img_num = len(gt_img_path_list)
for i in range(img_num):
    gt_img_path = gt_img_path_list[i]

    pred_img_path = pred_img_path_list[i]

    gt_img = tiff.imread(str(gt_img_path))
    pred_img = tiff.imread(str(pred_img_path))

    gt_img = gt_img.astype(np.float32) / 255.0
    pred_img = pred_img.astype(np.float32) / 255.0

    gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).unsqueeze(0)
    pred_img = torch.from_numpy(pred_img).unsqueeze(0)

    msg = f'evaluate {i + 1}/{img_num}'
    for metric in metric_list:
        metric_name = metric.__name__
        metric_value = metric(gt_img, pred_img)
        msg += f', {metric_name}: {metric_value:.4f}'
    txt_logger.info(msg)
