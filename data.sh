python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/split_size_256_stride_128 --tar_data_prefix syy_setting-1-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix syy_setting-1-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-2-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2 --tar_data_prefix syy_setting-2-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix syy_setting-3-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-3-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-4-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix syy_setting-4-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/split_size_256_stride_128 --tar_data_prefix syy_setting-5-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --tar_data_prefix syy_setting-5-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix syy_setting-6-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-6-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-7-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix syy_setting-7-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py;

python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/split_size_256_stride_128 --tar_data_prefix hh_setting-1-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix hh_setting-1-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2/split_size_256_stride_128 --tar_data_prefix hh_setting-2-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2 --tar_data_prefix hh_setting-2-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix hh_setting-3-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix hh_setting-3-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix hh_setting-4-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix hh_setting-4-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/split_size_256_stride_128 --tar_data_prefix hh_setting-5-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --tar_data_prefix hh_setting-5-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix hh_setting-6-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{} --tar_data_prefix hh_setting-6-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix hh_setting-7-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix hh_setting-7-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/hh_setting.py;
