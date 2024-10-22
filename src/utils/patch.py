from torch.nn.modules.utils import _pair, _quadruple


def cal_patch_num_hw(
    img_size,
    patch_size,
    patch_stride,
    is_drop_last=False,
):
    def _cal_patch_num_hw(
        img_size,
        patch_size,
        patch_stride,
        is_drop_last=False,
    ):
        is_divide_exactly = (img_size - patch_size) % patch_stride == 0
        if is_drop_last or is_divide_exactly:
            patch_num = (img_size - patch_size) // patch_stride + 1
        else:
            patch_num = (img_size - patch_size) // patch_stride + 2
        return patch_num

    return [
        _cal_patch_num_hw(img_size[i], patch_size[i], patch_stride[i], is_drop_last)
        for i in range(2)
    ]


def cal_padding_img_size(img_patch_num, patch_size, patch_stride):
    padding_img_size = [
        ((img_patch_num[i] - 1) * patch_stride[i] + patch_size[i]) for i in range(2)
    ]
    return padding_img_size


def cal_padding_img_pixel_num_hw(
    img_size, patch_size, patch_stride, is_drop_last=False
):
    img_patch_num = cal_patch_num_hw(img_size, patch_size, patch_stride, is_drop_last)
    padding_img_size = cal_padding_img_size(img_patch_num, patch_size, patch_stride)
    padding_img_pixel_num_top = (padding_img_size[0] - img_size[0]) // 2
    padding_img_pixel_num_bottom = (
        padding_img_size[0] - img_size[0] - padding_img_pixel_num_top
    )
    padding_img_pixel_num_left = (padding_img_size[1] - img_size[1]) // 2
    padding_img_pixel_num_right = (
        padding_img_size[1] - img_size[1] - padding_img_pixel_num_left
    )
    return (
        (padding_img_pixel_num_top, padding_img_pixel_num_bottom),
        (
            padding_img_pixel_num_left,
            padding_img_pixel_num_right,
        ),
    )


class PatchGenerator:
    patch_index = -1

    def __init__(self, img_size, patch_size, patch_stride):
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        patch_stride = _pair(patch_stride)
        assert (img_size[0] - patch_size[0]) % patch_stride[0] == 0 and (
            img_size[1] - patch_size[1]
        ) % patch_stride[1] == 0, 'img_size must be divisible'
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.img_patch_num_hw = cal_patch_num_hw(img_size, patch_size, patch_stride)
        self.img_patch_num = self.img_patch_num_hw[0] * self.img_patch_num_hw[1]

    @property
    def patch_h_index(self):
        h_index = self.patch_index // self.img_patch_num_hw[1]
        return h_index

    @property
    def patch_w_index(self):
        w_index = self.patch_index % self.img_patch_num_hw[1]
        return w_index

    def __next__(self):
        self.patch_index += 1
        if self.patch_index >= self.img_patch_num:
            raise StopIteration
        top = self.patch_h_index * self.patch_stride[0]
        bottom = top + self.patch_size[0]
        left = self.patch_w_index * self.patch_stride[1]
        right = left + self.patch_size[1]

        return (top, bottom, left, right)

    def __iter__(self):
        return self


# python -m src.utils.patch
if __name__ == '__main__':
    img_size = (512, 768)
    patch_size = (256, 256)
    patch_stride = (256, 256)
    patch_generator = PatchGenerator(img_size, patch_size, patch_stride)
    for patch_slice in patch_generator:
        print(
            patch_slice,
            patch_generator.patch_index,
            patch_generator.patch_h_index,
            patch_generator.patch_w_index,
        )
