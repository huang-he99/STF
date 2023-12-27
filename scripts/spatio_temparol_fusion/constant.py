# DATASET_TYPE = ['AHB', 'Daxing', 'Tianjin', 'CIA', 'LGC']
DATASET_TYPE = ['CIA', 'LGC']
SENSOR_TYPE = ['Landsat', 'MODIS']
BAND = {'AHB': 6, 'Daxing': 6, 'Tianjin': 6, 'CIA': 6, 'LGC': 6}

CROP_INFO = {
    'AHB': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'Daxing': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'Tianjin': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'CIA': {'is_crop': True, 'crop_shift': [20, 106], 'crop_size': [1792, 1280]},
    'LGC': {'is_crop': True, 'crop_shift': [40, 32], 'crop_size': [2560, 3072]},
}
