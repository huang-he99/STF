from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.stfgan import STFGANGenerator

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
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Resize(
        key_list=[
            'coarse_img_01_stage_1',
            'coarse_img_02_stage_1',
            'coarse_img_03_stage_1',
            'coarse_img_01_stage_2',
            'coarse_img_02_stage_2',
            'coarse_img_03_stage_2',
        ],
        scale_factor=1 / 4,
        interpolation_mode=0,
    ),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name='LGC',
    data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full',
    transform_func_list=test_transform_list,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

img_channel_num = 6

model_stage_1 = STFGANGenerator(img_channel_num=img_channel_num)
model_stage_2 = STFGANGenerator(img_channel_num=img_channel_num)

checkpoint_stage_1_path = 'results/stfgan/syy_setting-8/stage_1~RMSProp/checkpoints/model_generator_epoch_999.pth'
checkpoint_stage_2_path = 'results/stfgan/syy_setting-8/stage_2~RMSProp/checkpoints/model_generator_epoch_999.pth'

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
