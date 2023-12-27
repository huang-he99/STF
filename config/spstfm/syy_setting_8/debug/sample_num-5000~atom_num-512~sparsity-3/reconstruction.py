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
        fine_img_02='Landsat_02',
        fine_img_03='Landsat_03',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
        coarse_img_03='MODIS_03',
        extend_data=dict(
            dictionary_sparisty='',
        ),
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        fine_img_03='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
        coarse_img_03='{}_M_{}',
        extend_data=dict(
            dictionary_sparisty='{}_dictionary_sparisty_',
        ),
    ),
    data_suffix_dcit=dict(dictionary_sparisty='.mat'),
    is_serialize_data=True,
)

data_transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'fine_img_03',
    'coarse_img_01',
    'coarse_img_02',
    'coarse_img_03',
]

dictionary_sparisty_transforms_key_list = ['dictionary_sparisty']
np_key_list = ['sparisty_matrix', 'coarse_diff_dictionary', 'fine_diff_dictionary']
test_transform_list = [
    LoadData(key_list=data_transforms_key_list),
    RescaleToZeroOne(key_list=data_transforms_key_list, data_range=[0, 255]),
    LoadDictionarySparistyMatrix(
        key_list=dictionary_sparisty_transforms_key_list, np_key_list=np_key_list
    ),
    Format(key_list=data_transforms_key_list),
]

test_dataset = ConcatDataset(
    [
        dataset_cls_func(
            dataset_name='CIA',
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/test/full',
            extend_data_root='results/spstfm/syy_setting_8/sample_num-5000~atom_num-512~sparsity-3/training_dictionary_pair/dictionary_sparisty/CIA/',
            transform_func_list=test_transform_list,
        ),
        dataset_cls_func(
            dataset_name='LGC',
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/test/full',
            extend_data_root='results/spstfm/syy_setting_8/sample_num-5000~atom_num-512~sparsity-3/training_dictionary_pair/dictionary_sparisty/LGC/',
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


# val_transforms_list = [
#     LoadData(key_list=transforms_key_list),
#     RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 255]),
#     Resize(
#         key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
#         resize_shape=(64, 64),
#         interpolation_mode=0,
#         is_save_original_data=True,
#     ),
#     Format(
#         key_list=transforms_key_list
#         + ['ori_coarse_img_01', 'ori_coarse_img_02', 'ori_coarse_img_03']
#     ),
# ]

# val_dataset = ConcatDataset(
#     [
#         dataset_cls_func(
#             data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/val',
#             transform_func_list=val_transforms_list,
#         ),
#         dataset_cls_func(
#             data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/val',
#             transform_func_list=val_transforms_list,
#         ),
#     ]
# )

# val_dataloader = DataLoader(
#     dataset=val_dataset,
#     batch_size=1,
#     num_workers=1,
#     sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
#     worker_init_fn=partial(worker_init_fn, num_workers=1, rank=0, seed=42),
# )


patch_size = 7
sample_num = 5000
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

__all__ = ['test_dataloader', 'model', 'metric_list']
