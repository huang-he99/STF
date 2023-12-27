from src.utils.img.process.linear_stretch import truncated_linear_stretch
import argparse
import tifffile as tiff
from pathlib import Path
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from typing import Any


def linear_stretch_via_path(
    src_img_path,
    tar_data_dir_path,
    truncated_percent: int = 2,
    stretch_range: Any = [0, 255],
    is_drop_non_positive: bool = False,
):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_name = src_img_path.name
    tar_data_name = src_img_name
    tar_data_path = tar_data_dir_path / tar_data_name

    src_img = tiff.imread(src_img_path)
    if src_img.dtype == 'uint8':
        tar_data = src_img
    else:
        tar_data = truncated_linear_stretch(
            src_img, truncated_percent, stretch_range, is_drop_non_positive
        )
    return tar_data_path, tar_data


# python -m  scripts.spatio_temparol_fusion.process.linear_stretch --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --is_drop_non_positive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    parser.add_argument('--truncated_percent', type=int, default=2)
    parser.add_argument('--stretch_range', type=int, nargs='+', default=[0, 255])
    parser.add_argument('--is_drop_non_positive', action='store_true')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix = args.src_data_prefix
    tar_data_prefix = args.tar_data_prefix
    truncated_percent = args.truncated_percent
    stretch_range = args.stretch_range
    is_drop_non_positive = args.is_drop_non_positive

    print('linear_stretch')
    for dataset_type in DATASET_TYPE:
        for sensor_type in SENSOR_TYPE:
            src_data_dir_path = (
                root_path / dataset_type / src_data_prefix / f'original' / sensor_type
            )
            tar_data_dir_path = (
                root_path / dataset_type / tar_data_prefix / 'original' / sensor_type
            )
            tar_data_dir_path.mkdir(parents=True, exist_ok=True)
            src_data_path_list = list(src_data_dir_path.glob('*.tif'))
            pbar = tqdm(src_data_path_list)
            for data_index, src_data_path in enumerate(pbar):
                pbar.set_description(
                    f'format {dataset_type} {sensor_type}: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}'
                )
                tar_data_path, tar_data = linear_stretch_via_path(
                    src_data_path,
                    tar_data_dir_path,
                    truncated_percent,
                    stretch_range,
                    is_drop_non_positive,
                )
                tiff.imwrite(tar_data_path, tar_data)
