from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.swinstf import SwinSTFM

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

test_transform_list = [
    LoadData(key_list=transforms_key_list),
    Nan2Zero(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 1]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name='STIL',
    data_root='data/spatio_temporal_fusion/Mini_STIF_Dataset/STIL_Test',
    transform_func_list=test_transform_list,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

img_channel_num = 6

model = SwinSTFM()

checkpoint_path = 'results/swinstf/STIL/Adam_1e-4~StepLR_step_size-20_gamma_5e-1/checkpoints/model_epoch_49.pth'


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
    'test_dataloader',
    'model_stage_1',
    'model_stage_2',
    'checkpoint_stage_1_path',
    'checkpoint_stage_2_path',
    'optimizer',
    'scheduler',
    'metric_list',
]
