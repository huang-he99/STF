from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Union, Any
import copy
import re
import numpy as np
import pickle


class SpatioTemporalFusionDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        data_root: Union[str, Path],
        data_prefix_tmpl_dict: dict,
        data_name_tmpl_dict: dict,
        is_serialize_data: bool = False,
        transform_func_list: List = None,
    ):
        super(SpatioTemporalFusionDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.data_prefix_tmpl_dict = data_prefix_tmpl_dict
        self.data_name_tmpl_dict = data_name_tmpl_dict
        # self.search_key = 'fine_img_01'
        self.search_key = list(data_prefix_tmpl_dict.keys())[0]
        self.data_path_list = self.load_data_list()

        self.is_serialize_data = is_serialize_data
        if self.is_serialize_data:
            self.data_bytes, self.data_address = self.serialize_data()
        self.transform_func_list = transform_func_list

    def serialize_data(self):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_bytes_list = [_serialize(data) for data in self.data_path_list]
        data_size_list = np.asarray(
            [len(data_bytes) for data_bytes in data_bytes_list], dtype=np.int64
        )
        data_address = np.cumsum(data_size_list)
        data_types = np.concatenate(data_bytes_list)
        return data_types, data_address

    def load_data_list(self):
        data_path_segment_list = self.get_data_path_segment_list()
        data_path_list = []
        for (
            data_prefix_name_filled_context,
            data_file_name_filled_context,
            data_file_suffix,
        ) in data_path_segment_list:
            data_path_key = (
                '_'.join(data_prefix_name_filled_context)
                + '-'
                + '_'.join(data_file_name_filled_context)
            )
            data_path_dict = dict(key=data_path_key)
            for key in self.data_prefix_tmpl_dict.keys():
                data_prefix_tmpl = self.data_prefix_tmpl_dict[key]
                data_name_tmpl = self.data_name_tmpl_dict[key]
                data_prefix = data_prefix_tmpl.format(*data_prefix_name_filled_context)
                data_name = data_name_tmpl.format(*data_file_name_filled_context)
                path = self.data_root / data_prefix / f'{data_name}{data_file_suffix}'
                data_path_dict[f'{key}_path'] = str(path)
            data_path_list.append(data_path_dict)
        return data_path_list

    def get_data_path_segment_list(self):
        data_path_segment_list = []

        data_prefix_tmpl = self.data_prefix_tmpl_dict[self.search_key]
        data_prefix_regexp_for_pathlib = data_prefix_tmpl.replace('{}', '*')
        data_prefix_regexp_for_re = data_prefix_tmpl.replace('{}', '(.*)')

        data_name_tmpl = self.data_name_tmpl_dict[self.search_key]
        data_name_regexp_for_pathlib = data_name_tmpl.replace('{}', '*')
        data_name_regexp_for_re = data_name_tmpl.replace('{}', '(.*)')

        data_prefix_path_list = sorted(
            list(self.data_root.glob(data_prefix_regexp_for_pathlib))
        )
        for data_prefix_path in data_prefix_path_list:
            data_prefix_name = data_prefix_path.name
            data_prefix_name_filled_context = re.findall(
                data_prefix_regexp_for_re, data_prefix_name
            )[0]
            data_prefix_name_filled_context = (
                data_prefix_name_filled_context
                if isinstance(data_prefix_name_filled_context, tuple)
                else (data_prefix_name_filled_context,)
            )
            data_path_list = sorted(
                list(data_prefix_path.glob(data_name_regexp_for_pathlib))
            )
            for data_path in data_path_list:
                data_file_name, data_file_suffix = (
                    data_path.stem,
                    data_path.suffix,
                )
                data_file_name_filled_context = re.findall(
                    data_name_regexp_for_re, data_file_name
                )[0]
                data_file_name_filled_context = (
                    data_file_name_filled_context
                    if isinstance(data_file_name_filled_context, tuple)
                    else (data_file_name_filled_context,)
                )
                data_path_segment_list.append(
                    (
                        data_prefix_name_filled_context,
                        data_file_name_filled_context,
                        data_file_suffix,
                    )
                )
        return data_path_segment_list

    def __getitem__(self, idx: int) -> Any:
        if self.is_serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_path_list[idx])
        data_info['sample_idx'] = idx
        data_info['dataset_name'] = self.dataset_name
        for transform_func in self.transform_func_list:
            data_info = transform_func(data_info)
        return data_info

    def __len__(self) -> int:
        if self.is_serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_path_list)


class SpatioTemporalFusionDatasetForSPSTFM(Dataset):
    def __init__(
        self,
        dataset_name,
        data_root: Union[str, Path] = None,
        extend_data_root: Union[str, Path] = None,
        data_prefix_tmpl_dict: dict = {},
        data_name_tmpl_dict: dict = {},
        data_suffix_dcit: dict = {},
        is_serialize_data: bool = False,
        transform_func_list: List = None,
    ):
        super(SpatioTemporalFusionDatasetForSPSTFM, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.extend_data_root = (
            Path(extend_data_root) if extend_data_root is not None else None
        )
        self.data_prefix_tmpl_dict = data_prefix_tmpl_dict
        self.data_name_tmpl_dict = data_name_tmpl_dict
        self.data_suffix_dcit = data_suffix_dcit
        # self.search_key = 'fine_img_01'
        self.search_key = list(data_prefix_tmpl_dict.keys())[0]
        self.data_path_list = self.load_data_list()

        self.is_serialize_data = is_serialize_data
        if self.is_serialize_data:
            self.data_bytes, self.data_address = self.serialize_data()
        self.transform_func_list = transform_func_list

    def serialize_data(self):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_bytes_list = [_serialize(data) for data in self.data_path_list]
        data_size_list = np.asarray(
            [len(data_bytes) for data_bytes in data_bytes_list], dtype=np.int64
        )
        data_address = np.cumsum(data_size_list)
        data_types = np.concatenate(data_bytes_list)
        return data_types, data_address

    def load_data_list(self):
        data_path_segment_list = self.get_data_path_segment_list()
        data_path_list = []
        for (
            data_prefix_name_filled_context,
            data_file_name_filled_context,
            data_file_suffix,
        ) in data_path_segment_list:
            data_path_key = (
                '_'.join(data_prefix_name_filled_context)
                + '-'
                + '_'.join(data_file_name_filled_context)
            )
            data_path_dict = dict(key=data_path_key)
            for key in self.data_prefix_tmpl_dict.keys():
                if key == 'extend_data':
                    for extend_data_key in self.data_prefix_tmpl_dict[key].keys():
                        data_prefix_tmpl = self.data_prefix_tmpl_dict[key][
                            extend_data_key
                        ]
                        data_name_tmpl = self.data_name_tmpl_dict[key][extend_data_key]
                        data_prefix = data_prefix_tmpl.format(
                            *data_prefix_name_filled_context
                        )
                        data_name = data_name_tmpl.format(
                            *data_file_name_filled_context
                        )
                        preset_data_file_suffix = self.data_suffix_dcit.get(
                            extend_data_key, None
                        )
                        data_file_suffix = (
                            preset_data_file_suffix
                            if preset_data_file_suffix is not None
                            else data_file_suffix
                        )
                        path = (
                            self.extend_data_root
                            / data_prefix
                            / f'{data_name}{data_file_suffix}'
                        )
                        data_path_dict[f'{extend_data_key}_path'] = str(path)
                else:
                    data_prefix_tmpl = self.data_prefix_tmpl_dict[key]
                    data_name_tmpl = self.data_name_tmpl_dict[key]
                    data_prefix = data_prefix_tmpl.format(
                        *data_prefix_name_filled_context
                    )
                    data_name = data_name_tmpl.format(*data_file_name_filled_context)
                    preset_data_file_suffix = self.data_suffix_dcit.get(key, None)
                    data_file_suffix = (
                        preset_data_file_suffix
                        if preset_data_file_suffix is not None
                        else data_file_suffix
                    )
                    path = (
                        self.data_root / data_prefix / f'{data_name}{data_file_suffix}'
                    )
                    data_path_dict[f'{key}_path'] = str(path)
            data_path_list.append(data_path_dict)
        return data_path_list

    def get_data_path_segment_list(self):
        data_path_segment_list = []

        data_prefix_tmpl = self.data_prefix_tmpl_dict[self.search_key]
        data_prefix_regexp_for_pathlib = data_prefix_tmpl.replace('{}', '*')
        data_prefix_regexp_for_re = data_prefix_tmpl.replace('{}', '(.*)')

        data_name_tmpl = self.data_name_tmpl_dict[self.search_key]
        data_name_regexp_for_pathlib = data_name_tmpl.replace('{}', '*')
        data_name_regexp_for_re = data_name_tmpl.replace('{}', '(.*)')

        data_prefix_path_list = sorted(
            list(self.data_root.glob(data_prefix_regexp_for_pathlib))
        )
        for data_prefix_path in data_prefix_path_list:
            data_prefix_name = data_prefix_path.name
            data_prefix_name_filled_context = re.findall(
                data_prefix_regexp_for_re, data_prefix_name
            )[0]
            data_prefix_name_filled_context = (
                data_prefix_name_filled_context
                if isinstance(data_prefix_name_filled_context, tuple)
                else (data_prefix_name_filled_context,)
            )
            data_path_list = sorted(
                list(data_prefix_path.glob(data_name_regexp_for_pathlib))
            )
            for data_path in data_path_list:
                data_file_name, data_file_suffix = (
                    data_path.stem,
                    data_path.suffix,
                )
                data_file_name_filled_context = re.findall(
                    data_name_regexp_for_re, data_file_name
                )[0]
                data_file_name_filled_context = (
                    data_file_name_filled_context
                    if isinstance(data_file_name_filled_context, tuple)
                    else (data_file_name_filled_context,)
                )
                data_path_segment_list.append(
                    (
                        data_prefix_name_filled_context,
                        data_file_name_filled_context,
                        data_file_suffix,
                    )
                )
        return data_path_segment_list

    def __getitem__(self, idx: int) -> Any:
        if self.is_serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_path_list[idx])
        data_info['sample_idx'] = idx
        data_info['dataset_name'] = self.dataset_name
        for transform_func in self.transform_func_list:
            data_info = transform_func(data_info)
        return data_info

    def __len__(self) -> int:
        if self.is_serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_path_list)


# python -m src.data.dataset
if __name__ == '__main__':
    from torch.utils.data import DataLoader, ConcatDataset
    from src.data.dataloader.data_sampler import EpochBasedSampler
    from src.data.dataloader.worker_init import worker_init_fn
    from functools import partial
    from src.data.transforms import *

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

    train_transforms_key_list = [
        'fine_img_01',
        'fine_img_02',
        'fine_img_03',
        'coarse_img_01',
        'coarse_img_02',
        'coarse_img_03',
    ]

    train_transforms = [
        LoadData(key_list=train_transforms_key_list),
        RescaleToMinusOneOne(key_list=train_transforms_key_list, data_range=[0, 10000]),
        Rotate(key_list=train_transforms_key_list),
        Flip(key_list=train_transforms_key_list),
        Resize(
            key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
            resize_shape=(64, 64),
            interpolation_mode=0,
            is_save_original_data=False,
            is_remain_original_data=False,
        ),
        Format(key_list=train_transforms_key_list),
    ]

    train_dataset = ConcatDataset(
        [
            dataset_cls_func(
                data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-1-patch/train',
                transform_func_list=train_transforms,
            ),
            dataset_cls_func(
                data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-1-patch/train',
                transform_func_list=train_transforms,
            ),
        ]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
        num_workers=4,
        worker_init_fn=partial(worker_init_fn, num_workers=4, rank=0, seed=42),
    )

    for batch_idx, batch_data in enumerate(train_dataloader):
        print(batch_idx, batch_data)
