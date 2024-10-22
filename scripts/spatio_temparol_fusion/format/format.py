from pathlib import Path
from scripts.spatio_temparol_fusion.format import format_file_name, format_data
import tifffile as tiff
import argparse
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm


def format(src_data_path, tar_data_dir_path, band):
    src_data_path = Path(src_data_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_data_name = src_data_path.name
    tar_data_name = format_file_name(src_data_name)
    tar_data_path = tar_data_dir_path / tar_data_name

    src_data = tiff.imread(src_data_path)
    tar_data = format_data(src_data, band)
    return tar_data_path, tar_data


# python -m scripts.spatio_temparol_fusion.format.format --root_path data/spatio_temporal_fusion --src_data_prefix raw_data --tar_data_prefix format_data
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix = args.src_data_prefix
    tar_data_prefix = args.tar_data_prefix

    print('format')

    for dataset_type in DATASET_TYPE:
        for sensor_type in SENSOR_TYPE:
            src_data_dir_path = root_path / dataset_type / src_data_prefix / sensor_type
            tar_data_dir_path = (
                root_path
                / dataset_type
                / f'public_processing_data'
                / tar_data_prefix
                / 'original'
                / sensor_type
            )
            tar_data_dir_path.mkdir(parents=True, exist_ok=True)
            src_data_path_list = list(src_data_dir_path.glob('*.tif'))
            pbar = tqdm(src_data_path_list)
            for data_index, src_data_path in enumerate(pbar):
                pbar.set_description(
                    f'format {dataset_type} {sensor_type}: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}'
                )
                tar_data_path, tar_data = format(
                    src_data_path, tar_data_dir_path, band=BAND[dataset_type]
                )
                tiff.imwrite(tar_data_path, tar_data)
