from functools import partial

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.stfgan import STFGANDiscriminator, STFGANGenerator


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        fine_img_03='Landsat_03',
        coarse_img_01='Landsat_01',
        coarse_img_02='Landsat_02',
        coarse_img_03='Landsat_03',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        fine_img_03='{}_L_{}',
        coarse_img_01='{}_L_{}',
        coarse_img_02='{}_L_{}',
        coarse_img_03='{}_L_{}',
    ),
    is_serialize_data=True,
)


transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'fine_img_03',
    'coarse_img_01',
    'coarse_img_02',
    'coarse_img_03',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Resize(
        key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
        resize_shape=(64, 64),
        interpolation_mode=0,
        is_save_original_data=False,
        is_remain_original_data=False,
    ),
    Format(key_list=transforms_key_list),
]

train_dataset = ConcatDataset(
    [
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train',
            transform_func_list=train_transform_list,
        ),
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/train',
            transform_func_list=train_transform_list,
        ),
    ]
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=4,
    worker_init_fn=partial(worker_init_fn, num_workers=1, rank=0, seed=42),
)


val_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Resize(
        key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
        resize_shape=(64, 64),
        interpolation_mode=0,
        is_save_original_data=True,
    ),
    Format(
        key_list=transforms_key_list
        + ['ori_coarse_img_01', 'ori_coarse_img_02', 'ori_coarse_img_03']
    ),
]

val_dataset = ConcatDataset(
    [
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/val',
            transform_func_list=val_transforms_list,
        ),
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/val',
            transform_func_list=val_transforms_list,
        ),
    ]
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=1,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
    worker_init_fn=partial(worker_init_fn, num_workers=1, rank=0, seed=42),
)

test_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Resize(
        key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
        resize_shape=(64, 64),
        interpolation_mode=0,
        is_save_original_data=True,
    ),
    Format(
        key_list=transforms_key_list
        + ['ori_coarse_img_01', 'ori_coarse_img_02', 'ori_coarse_img_03']
    ),
]
test_dataset = ConcatDataset(
    [
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full',
            transform_func_list=test_transforms_list,
        ),
        dataset_cls_func(
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full',
            transform_func_list=test_transforms_list,
        ),
    ]
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    worker_init_fn=partial(worker_init_fn, num_workers=1, rank=0, seed=42),
)

model_generator = STFGANGenerator(img_channel_num=6)
model_discriminator = STFGANDiscriminator(img_channel_num=6)

optimizer_generator = partial(torch.optim.RMSprop, lr=1e-4)
optimizer_discriminator = partial(torch.optim.RMSprop, lr=1e-4)

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

__all__ = [
    'train_dataloader',
    'val_dataloader',
    'model_generator',
    'model_discriminator',
    'optimizer_generator',
    'optimizer_discriminator',
    'metric_list',
]
