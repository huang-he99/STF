import sys
from src.data.dataset import SpatioTemporalFusionDataset
from accelerate import Accelerator
from torch.utils.data import DataLoader, ConcatDataset
from src.data.dataloader.data_sampler import EpochBasedSampler
from functools import partial
from src.data.transforms import *
from src.data.dataloader.worker_init import worker_init_fn

accelerator = Accelerator()

dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img='Landsat_{}',
        coarse_img='MODIS_{}',
    ),
    data_name_tmpl_dict=dict(
        fine_img='{}_L_{}',
        coarse_img='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img',
    'coarse_img',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

train_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/train',
    transform_func_list=train_transform_list,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=4,
)

train_dataloader = accelerator.prepare(train_dataloader)
for batch_idx, data in enumerate(train_dataloader):
    print(batch_idx)
    print(data)
    break
