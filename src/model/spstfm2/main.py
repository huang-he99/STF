from pathlib import Path
from skimage import io
import numpy as np
import torch
import spstfm

if __name__ == '__main__':
    L1_path = Path(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/3BandsRGB/Split/CIA/Landsat/2001_10_08/L_2001-10-08_008.tif'
    )
    L3_path = Path(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/3BandsRGB/Split/CIA/Landsat/2001_10_17/L_2001-10-17_008.tif'
    )

    M1_path = Path(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/3BandsRGB/Split/CIA/MODIS/2001_10_08/M_2001-10-08_008.tif'
    )
    M3_path = Path(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/3BandsRGB/Split/CIA/MODIS/2001_10_17/M_2001-10-17_008.tif'
    )

    L1_img = io.imread(L1_path).transpose(2, 0, 1)
    L3_img = io.imread(L3_path).transpose(2, 0, 1)
    M1_img = io.imread(M1_path).transpose(2, 0, 1)
    M3_img = io.imread(M3_path).transpose(2, 0, 1)

    L1_img = torch.from_numpy(L1_img).float().unsqueeze(0)
    L3_img = torch.from_numpy(L3_img).float().unsqueeze(0)
    M1_img = torch.from_numpy(M1_img).float().unsqueeze(0)
    M3_img = torch.from_numpy(M3_img).float().unsqueeze(0)

    L1_img_single_band = L1_img[:, 0:1, :, :]
    L3_img_single_band = L3_img[:, 0:1, :, :]
    M1_img_single_band = M1_img[:, 0:1, :, :]
    M3_img_single_band = M3_img[:, 0:1, :, :]

    L1_img_single_band = L1_img_single_band / 255.0
    L3_img_single_band = L3_img_single_band / 255.0
    M1_img_single_band = M1_img_single_band / 255.0
    M3_img_single_band = M3_img_single_band / 255.0

    spstfm_model = spstfm.SPSTFM(
        sample_dim=49,
        sample_num=2000,
        atom_num=1024,
        max_iter=30,
        init_method='data_elements',
        patch_size=7,
        stride=3,
        sparsity=5,
    )

    (
        coarse_dictionary,
        fine_dictionary,
        sparsity_matrix,
    ) = spstfm_model.training_dictionary_pair_per_channel(
        M1_img_single_band, M3_img_single_band, L1_img_single_band, L3_img_single_band
    )

    coarse_dictionary = coarse_dictionary.squeeze().cpu().numpy()
    fine_dictionary = fine_dictionary.squeeze().cpu().numpy()
    sparsity_matrix = sparsity_matrix.squeeze().cpu().numpy()

    c_img = coarse_dictionary @ sparsity_matrix
    f_img = fine_dictionary @ sparsity_matrix

    import matplotlib.pyplot as plt

    plt.imshow(c_img.squeeze(), cmap='gray')
    plt.show()
    plt.imshow(f_img.squeeze(), cmap='gray')
    plt.show()
    pass
