from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.sr3 import GaussianDiffusion, UNet


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img="L2",
        coarse_img="M2",
    ),
    data_name_tmpl_dict=dict(
        fine_img="{}",
        coarse_img="{}",
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    "fine_img",
    "coarse_img",
]


test_transforms_list = [
    LoadData(key_list=transforms_key_list),
    Nan2Zero(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 1]),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name="STIL",
    data_root="data/spatio_temporal_fusion/Mini_STIF_Dataset/STIL_Test",
    transform_func_list=test_transforms_list,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
)


model = GaussianDiffusion(
    denoise_fn=UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=16,
        channel_mults=[1, 2, 4, 8, 16],
        attn_res=[],
        res_blocks=1,
        dropout=0,
        # image_size=512,
    ),
    image_size=512,
    channels=3,
    conditional=True,
    schedule_opt={
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 1e-2,
    },
)

checkpoint_path = "/home/hh/container/Pretrain_Model/sr3/I830000_E32_gen.pth"


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
    "test_dataloader",
    "model",
    "checkpoint_path",
    "metric_list",
]
