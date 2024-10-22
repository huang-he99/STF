import numpy as np


def truncated_linear_stretch(
    image, truncated_percent=2, stretch_range=[0, 255], is_drop_non_positive=False
):
    """_summary_

    Args:
        image (np.array): HWC or HW
        truncated_percent (int, optional): _description_. Defaults to 2.
        stretch_range (list, optional): _description_. Defaults to [0, 255].
    """
    max_tansformed_img = (
        np.where(image <= 0, 65536, image) if is_drop_non_positive else image
    )
    min_tansformed_img = (
        np.where(image <= 0, -65536, image) if is_drop_non_positive else image
    )

    truncated_lower = np.percentile(
        max_tansformed_img, truncated_percent, axis=(0, 1), keepdims=True
    )
    truncated_upper = np.percentile(
        min_tansformed_img, 100 - truncated_percent, axis=(0, 1), keepdims=True
    )

    stretched_img = (image - truncated_lower) / (truncated_upper - truncated_lower) * (
        stretch_range[1] - stretch_range[0]
    ) + stretch_range[0]
    stretched_img[stretched_img < stretch_range[0]] = stretch_range[0]
    stretched_img[stretched_img > stretch_range[1]] = stretch_range[1]
    if stretch_range[1] <= 255:
        stretched_img = np.uint8(stretched_img)
    elif stretch_range[1] <= 65535:
        stretched_img = np.uint16(stretched_img)
    return stretched_img
