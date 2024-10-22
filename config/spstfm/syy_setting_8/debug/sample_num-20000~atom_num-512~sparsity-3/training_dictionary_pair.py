from functools import partial
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import (
    SpatioTemporalFusionDatasetForSPSTFM as SpatioTemporalFusionDataset,
)
from src.data.transforms import *
from src.metrics import *
from src.model.spstfm import SPSTFM


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_03='Landsat_03',
        coarse_img_01='MODIS_01',
        coarse_img_03='MODIS_03',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_03='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_03='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01',
    'fine_img_03',
    'coarse_img_01',
    'coarse_img_03',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToZeroOne(key_list=transforms_key_list, data_range=[0, 255]),
    Format(key_list=transforms_key_list),
]

train_dataset = ConcatDataset(
    [
        dataset_cls_func(
            dataset_name='CIA',
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/test/full',
            transform_func_list=train_transform_list,
        ),
        dataset_cls_func(
            dataset_name='LGC',
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/test/full',
            transform_func_list=train_transform_list,
        ),
    ]
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

patch_size = 7
sample_num = 20000
atom_num = 512
max_iter = 100
sparsity = 3
stride = 3

model = SPSTFM(
    sample_dim=patch_size**2,
    sample_num=sample_num,
    atom_num=atom_num,
    patch_size=patch_size,
    max_iter=max_iter,
    sparsity=sparsity,
    stride=stride,
)

# metric_list = [
#     RMSE(),
#     MAE(),
#     PSNR(max_value=1.0),
#     SSIM(data_range=1.0),
#     ERGAS(ratio=1.0 / 16.0),
#     CC(),
#     SAM(),
#     UIQI(data_range=1.0),
# ]

__all__ = [
    'train_dataloader',
    'model',
]
