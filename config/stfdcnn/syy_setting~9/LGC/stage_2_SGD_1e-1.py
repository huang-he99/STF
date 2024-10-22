from functools import partial

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.stfdcnn import STFDCNN


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img='Landsat_{}',
        coarse_img='Landsat_{}',
    ),
    data_name_tmpl_dict=dict(
        fine_img='{}_L_{}',
        coarse_img='{}_L_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img',
    'coarse_img',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000.0]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Resize(
        key_list=['coarse_img'],
        resize_shape=(64, 64),
        interpolation_mode=0,
        is_save_original_data=False,
        is_remain_original_data=True,
    ),
    Format(key_list=transforms_key_list),
]

train_dataset = dataset_cls_func(
    dataset_name='LGC',
    data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/train',
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
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000.0]),
    Resize(
        key_list=['coarse_img'],
        resize_shape=(64, 64),
        interpolation_mode=0,
        is_save_original_data=False,
        is_remain_original_data=True,
    ),
    Format(key_list=transforms_key_list),
]

val_dataset = dataset_cls_func(
    dataset_name='LGC',
    data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/val',
    transform_func_list=val_transforms_list,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)


img_channel_num = 6

model = STFDCNN(input_channels=img_channel_num, out_channels=img_channel_num)

optimizer = partial(torch.optim.SGD, lr=1e-1)
scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=1000, gamma=0.1)


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

# device_info = DeviceInfo(cuda_idx=0, is_auto_empty_cache=True)
# device_info =

__all__ = [
    'train_dataloader',
    'val_dataloader',
    'model',
    'optimizer',
    'scheduler',
    'metric_list',
]
