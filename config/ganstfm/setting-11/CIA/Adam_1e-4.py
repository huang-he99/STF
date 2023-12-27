from functools import partial
import sched

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataloader.data_sampler import EpochBasedSampler

from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.ganstfm import SFFusion, MSDiscriminator


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

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToZeroOne(key_list=transforms_key_list, data_range=[0, 10000]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

train_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train',
    transform_func_list=train_transform_list,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=4,
)


val_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToZeroOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
]

val_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/val',
    transform_func_list=val_transforms_list,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)


model_generator = SFFusion()
model_discriminator = MSDiscriminator()

optimizer_generator = partial(torch.optim.Adam, lr=1e-4)
optimizer_discriminator = partial(torch.optim.Adam, lr=1e-4)


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

w_change = 1e-2

__all__ = [
    'train_dataloader',
    'val_dataloader',
    'model_generator',
    'model_discriminator',
    'optimizer_generator',
    'optimizer_discriminator',
    'metric_list',
]
