### setting-1

#### 反射率数据 patch_size_256_stride_128

raw -> format -> split_size_256_stride_128

```bash
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/split_size_256_stride_128 --tar_data_prefix syy_setting-1-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 反射率数据 full

raw -> format

```bash
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix syy_setting-1-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-2

#### 假彩色(4-3-2) 反射率数据 patch_size_256_stride_128

raw -> format -> band_4-3-2 -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-2-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 假彩色(4-3-2)反射率数据 full

raw -> format -> band_4-3-2

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2 --tar_data_prefix syy_setting-2-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-3

#### crop 反射率数据 patch_size_256_stride_128

raw -> format -> crop_{}\_{}\_{}\_{} -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix syy_setting-3-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### crop 反射率数据 full

raw -> format -> crop_{}\_{}\_{}\_{}

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-3-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-4

#### crop 反射率数据 假彩色(4-3-2) patch_size_256_stride_128

raw -> format -> crop_{}\_{}\_{}\_{} -> band_4-3-2 -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-4-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### crop 反射率数据 假彩色(4-3-2) full

raw -> format -> crop_{}\_{}\_{}\_{} -> band_4-3-2

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix syy_setting-4-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-5

#### 线性拉伸(0~255))  patch_size_256_stride_128

raw -> format -> linear_stretch_percent_2 -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/split_size_256_stride_128 --tar_data_prefix syy_setting-5-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 线性拉伸(0~255)) full

raw -> format -> linear_stretch_percent_2

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --tar_data_prefix syy_setting-5-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-6

#### 线性拉伸(0~255)) crop  patch_size_256_stride_128

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{} -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/split_size_256_stride_128 --tar_data_prefix syy_setting-6-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 线性拉伸(0~255)) crop full

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{}

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-6-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-7

#### 线性拉伸(0~255)) crop 假彩色(4-3-2)  patch_size_256_stride_128

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{} -> band_4-3-2 -> split_size_256_stride_128

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_128 --tar_data_prefix syy_setting-7-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 线性拉伸(0~255)) crop 假彩色(4-3-2) full

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{} -> band_4-3-2

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix syy_setting-7-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-8

#### 线性拉伸(0~255)) crop 假彩色(4-3-2)  patch_size_256_stride_256

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{} -> band_4-3-2 -> split_size_256_stride_256

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_256_stride_256 --tar_data_prefix syy_setting-8-patch --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### 线性拉伸(0~255)) crop 假彩色(4-3-2) full

raw -> format -> linear_stretch_percent_2 -> crop_{}\_{}\_{}\_{} -> band_4-3-2

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix syy_setting-8-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-9

#### crop 反射率数据 patch_size_256_stride_256

raw -> format -> crop_{}\_{}\_{}\_{} -> split_size_256_stride_256

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_256_stride_256 --tar_data_prefix syy_setting-9 --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

#### crop 反射率数据 full

raw -> format -> crop_{}\_{}\_{}\_{}

```python
python -m scripts.spatio_temparol_fusion.dataset_generation.data_generation --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix syy_setting-9-full --dataset_setting_congfig_path scripts/spatio_temparol_fusion/dataset_generation/dataset_config/syy_setting.py
```

### setting-4

## Shell

```shell
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

```
