from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.stfdiff.model6 import PredNoiseNet, GaussianDiffusion


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'coarse_img_01',
    'coarse_img_02',
]

test_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full',
    transform_func_list=test_transform_list,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=True, seed=42),
    num_workers=0,
)


model = GaussianDiffusion(
    model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    image_size=256,
    timesteps=100,
    sampling_timesteps=10,
    objective='pred_x0',
    ddim_sampling_eta=1.0,
)

checkpoint_path = 'results1/stfdiff/syy_setting-9/model6/CIA/Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/checkpoints/model_epoch_200.pth'

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(),
]

__all__ = [
    'test_dataloader',
    'checkpoint_path',
    'model',
    'metric_list',
]
