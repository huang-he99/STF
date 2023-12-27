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
        fine_img_01_stage_1='Landsat_01',
        fine_img_02_stage_1='Landsat_02',
        fine_img_03_stage_1='Landsat_03',
        coarse_img_01_stage_1='MODIS_01',
        coarse_img_02_stage_1='MODIS_02',
        coarse_img_03_stage_1='MODIS_03',
        fine_img_01_stage_2='Landsat_01',
        fine_img_02_stage_2='Landsat_02',
        fine_img_03_stage_2='Landsat_03',
        coarse_img_01_stage_2='Landsat_01',
        coarse_img_02_stage_2='Landsat_02',
        coarse_img_03_stage_2='Landsat_03',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01_stage_1='{}_L_{}',
        fine_img_02_stage_1='{}_L_{}',
        fine_img_03_stage_1='{}_L_{}',
        coarse_img_01_stage_1='{}_M_{}',
        coarse_img_02_stage_1='{}_M_{}',
        coarse_img_03_stage_1='{}_M_{}',
        fine_img_01_stage_2='{}_L_{}',
        fine_img_02_stage_2='{}_L_{}',
        fine_img_03_stage_2='{}_L_{}',
        coarse_img_01_stage_2='{}_L_{}',
        coarse_img_02_stage_2='{}_L_{}',
        coarse_img_03_stage_2='{}_L_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01_stage_1',
    'fine_img_02_stage_1',
    'fine_img_03_stage_1',
    'coarse_img_01_stage_1',
    'coarse_img_02_stage_1',
    'coarse_img_03_stage_1',
    'fine_img_01_stage_2',
    'fine_img_02_stage_2',
    'fine_img_03_stage_2',
    'coarse_img_01_stage_2',
    'coarse_img_02_stage_2',
    'coarse_img_03_stage_2',
]

test_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 255]),
    Resize(
        key_list=[
            'fine_img_01_stage_1',
            'fine_img_02_stage_1',
            'fine_img_03_stage_1',
            'coarse_img_01_stage_2',
            'coarse_img_02_stage_2',
            'coarse_img_03_stage_2',
        ],
        scale_factor=1 / 8,
        interpolation_mode=0,
        is_save_original_data=False,
        is_remain_original_data=True,
    ),
    Format(key_list=transforms_key_list),
]

test_dataset = ConcatDataset(
    [
        dataset_cls_func(
            dataset_name='CIA',
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/test/full',
            transform_func_list=test_transform_list,
        ),
        dataset_cls_func(
            dataset_name='LGC',
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/test/full',
            transform_func_list=test_transform_list,
        ),
    ]
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

model_stage_1 = STFDCNN(input_channels=3, out_channels=3)
model_stage_2 = STFDCNN(input_channels=3, out_channels=3)
checkpoint_stage_1_path = (
    'results/stfdcnn/syy_setting-8/stage_1_SGD_1e-2/checkpoints/model_epoch_49.pth'
)
checkpoint_stage_2_path = (
    'results/stfdcnn/syy_setting-8/stage_2_SGD_1e-1/checkpoints/model_epoch_49.pth'
)

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=2047.0 / 255.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(data_range=1.0),
]

# device_info = DeviceInfo(cuda_idx=0, is_auto_empty_cache=True)
# device_info =

__all__ = [
    'test_dataloader',
    'model_stage_1',
    'model_stage_2',
    'checkpoint_stage_1_path',
    'checkpoint_stage_2_path',
    'optimizer',
    'scheduler',
    'metric_list',
]
