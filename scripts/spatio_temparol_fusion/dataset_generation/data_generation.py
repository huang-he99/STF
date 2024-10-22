from pathlib import Path
from scripts.spatio_temparol_fusion.format import format_file_name, format_data
import tifffile as tiff
import argparse
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from copy import deepcopy
import shutil

# python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/split_size_256_stride_128 --tar_data_prefix syy_setting-1-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py

# python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix syy_setting-1-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py

# python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/split_size_256_stride_128 --tar_data_prefix hh_setting-1-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py

# python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix hh_setting-1-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generation')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    parser.add_argument('--dataset_setting_congfig_path', type=str, required=True)
    args = parser.parse_args()

    print('data_generation')

    root_path = Path(args.root_path)
    src_data_prefix_tmpl = args.src_data_prefix
    tar_data_prefix = args.tar_data_prefix
    dataset_setting_congfig_path = args.dataset_setting_congfig_path
    # import importlib
    # dataset_setting_config_module = importlib.import_module(
    #     dataset_setting_congfig_path.replace('/', '.').replace('.py', '')
    # )
    dataset_setting_config_module_string = dataset_setting_congfig_path.replace(
        '/', '.'
    ).replace('.py', '')
    assert (
        dataset_setting_config_module_string.split('.')[-1]
        == tar_data_prefix.split('-')[0]
    ), f'setting module name{dataset_setting_config_module_string} and tar data file name{tar_data_prefix} should be same'
    exec(f'from {dataset_setting_config_module_string} import *')
    # import scripts.spatio_temparol_fusion.dataset_generation.dataset_config.hh_setting as hh_setting

    # d = hh_setting.DATASET_SETTING
    dataset_setting = deepcopy(DATASET_SETTING)  # type: ignore
    phase_list = ['train', 'val', 'test']
    for dataset_type in DATASET_TYPE:
        if 'crop' in src_data_prefix_tmpl:
            crop_info = CROP_INFO[dataset_type]
            crop_shift = crop_info['crop_shift']
            crop_size = crop_info['crop_size']
            crop_top = crop_shift[0]
            crop_bottom = crop_top + crop_size[0]
            crop_left = crop_shift[1]
            crop_right = crop_left + crop_size[1]
            src_data_prefix = src_data_prefix_tmpl.format(
                crop_top, crop_bottom, crop_left, crop_right
            )
        else:
            src_data_prefix = src_data_prefix_tmpl
        if DATASET_SETTING[dataset_type] == {}:  # type: ignore
            continue
        for phase in phase_list:
            data_group_list = dataset_setting[dataset_type].get(phase, None)
            if data_group_list is None:
                if phase != 'test':
                    raise ValueError(f'{dataset_type} {phase} data_group_list is None')
                else:
                    data_group_list = dataset_setting[dataset_type]['val']
            for sensor_type in SENSOR_TYPE:
                src_data_dir_path = (
                    root_path
                    / dataset_type
                    / src_data_prefix
                    / 'original'
                    / sensor_type
                )
                tar_data_group_dir_path = (
                    root_path / dataset_type / f'private_data' / tar_data_prefix / phase
                )
                if phase == 'test':
                    tar_data_group_dir_path = (
                        root_path
                        / dataset_type
                        / f'private_data'
                        / tar_data_prefix
                        / phase
                        / 'patch'
                    )
                pbar = tqdm(data_group_list)
                for data_group_idx, data_group in enumerate(pbar):
                    data_pair_idx = 0
                    for data_date in data_group:
                        data_pair_idx += 1
                        tar_data_dir_path = (
                            tar_data_group_dir_path
                            / f'{sensor_type}_{data_pair_idx:0>2d}'
                        )
                        tar_data_dir_path.mkdir(parents=True, exist_ok=True)

                        src_data_path_list = list(
                            src_data_dir_path.glob(f'*{data_date}*.tif')
                        ) + list(src_data_dir_path.glob(f'*{data_date}*/*.tif'))
                        for src_data_path in src_data_path_list:
                            src_stem, suffix = src_data_path.stem, src_data_path.suffix
                            tar_stem = src_stem.split(data_date)[0]
                            tar_stem = f'Group_{data_group_idx+1:0>2d}_{tar_stem}'
                            tar_data_path = tar_data_dir_path / f'{tar_stem}{suffix}'
                            pbar.set_description(
                                f'{dataset_type} {phase} {sensor_type} data_group_idx:{data_group_idx+1:02d}'
                            )
                            shutil.copy(src_data_path, tar_data_path)
                if phase == 'test':
                    src_data_dir_path = (
                        (root_path / dataset_type / src_data_prefix).parent
                        / 'original'
                        / sensor_type
                    )

                    tar_data_group_dir_path = (
                        root_path
                        / dataset_type
                        / f'private_data'
                        / tar_data_prefix
                        / phase
                        / 'full'
                    )
                    pbar = tqdm(data_group_list)

                    for data_group_idx, data_group in enumerate(pbar):
                        data_pair_idx = 0
                        for data_date in data_group:
                            data_pair_idx += 1
                            tar_data_dir_path = (
                                tar_data_group_dir_path
                                / f'{sensor_type}_{data_pair_idx:0>2d}'
                            )
                            tar_data_dir_path.mkdir(parents=True, exist_ok=True)

                            src_data_path_list = list(
                                src_data_dir_path.glob(f'*{data_date}*.tif')
                            ) + list(src_data_dir_path.glob(f'*{data_date}*/*.tif'))
                            for src_data_path in src_data_path_list:
                                src_stem, suffix = (
                                    src_data_path.stem,
                                    src_data_path.suffix,
                                )
                                tar_stem = src_stem.split(data_date)[0]
                                tar_stem = f'Group_{data_group_idx+1:0>2d}_{tar_stem}'
                                tar_data_path = (
                                    tar_data_dir_path / f'{tar_stem}{suffix}'
                                )
                                pbar.set_description(
                                    f'{dataset_type} {phase} {sensor_type} data_group_idx:{data_group_idx+1:02d}'
                                )
                                shutil.copy(src_data_path, tar_data_path)
