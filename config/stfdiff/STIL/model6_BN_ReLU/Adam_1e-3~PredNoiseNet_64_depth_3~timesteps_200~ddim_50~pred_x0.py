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
        fine_img_01='L1',
        fine_img_02='L2',
        coarse_img_01='M1',
        coarse_img_02='M2',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}',
        fine_img_02='{}',
        coarse_img_01='{}',
        coarse_img_02='{}',
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
    Nan2Zero(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 1]),
    Format(key_list=transforms_key_list),
]

train_dataset = dataset_cls_func(
    dataset_name='STIL',
    data_root='data/spatio_temporal_fusion/Mini_STIF_Dataset_split_size_256_stride_256/STIL_Train',
    transform_func_list=train_transform_list,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=24,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=8,
)


val_transforms_list = [
    LoadData(key_list=transforms_key_list),
    Nan2Zero(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 1]),
    Format(key_list=transforms_key_list),
]

val_dataset = dataset_cls_func(
    dataset_name='STIL',
    data_root='data/spatio_temporal_fusion/Mini_STIF_Dataset/STIL_Test',
    transform_func_list=val_transforms_list,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=24,
    num_workers=8,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)


model = GaussianDiffusion(
    model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    image_size=256,
    timesteps=200,
    sampling_timesteps=50,
    objective='pred_x0',
    ddim_sampling_eta=0.0,
)

optimizer = partial(torch.optim.Adam, lr=1e-3)


scheduler = None


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
    'train_dataloader',
    'val_dataloader',
    'model',
    'optimizer',
    'scheduler',
    'metric_list',
]
