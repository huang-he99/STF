from torch import nn
import torch


class DCNN(nn.Module):
    def __init__(
        self,
        input_channels=6,
        feature_extration_channels=32,
        non_linear_channels=16,
        feature_extration_kernel_size=9,
        non_linear_mapping_kernel_size=5,
        reconstruction_kernel_size=5,
        output_channels=6,
    ):
        super().__init__()
        self.feature_extraction_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=feature_extration_channels,
                kernel_size=feature_extration_kernel_size,
                padding="same",
            ),
            nn.ReLU(inplace=True),
        )
        self.nonlinear_mapping_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_extration_channels,
                out_channels=non_linear_channels,
                kernel_size=non_linear_mapping_kernel_size,
                padding="same",
            ),
            nn.ReLU(inplace=True),
        )
        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=non_linear_channels,
                out_channels=output_channels,
                kernel_size=reconstruction_kernel_size,
                padding="same",
            )
        )

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01):
        coarse_img_diff = coarse_img_02 - coarse_img_01
        input = torch.cat((coarse_img_diff, fine_img_01), dim=1)
        fine_img_diff = self.reconstruction_layer(
            self.nonlinear_mapping_layer(self.feature_extraction_layer(input))
        )
        return fine_img_diff


class STFNet(nn.Module):
    def __init__(
        self,
        input_channels=6,
        feature_extration_channels=32,
        non_linear_channels=16,
        feature_extration_kernel_size=9,
        non_linear_mapping_kernel_size=5,
        reconstruction_kernel_size=5,
        output_channels=6,
    ):
        super().__init__()
        self.DCNN_mapping_1 = DCNN(
            input_channels=input_channels * 2,
            feature_extration_channels=feature_extration_channels,
            non_linear_channels=non_linear_channels,
            feature_extration_kernel_size=feature_extration_kernel_size,
            non_linear_mapping_kernel_size=non_linear_mapping_kernel_size,
            reconstruction_kernel_size=reconstruction_kernel_size,
            output_channels=output_channels,
        )
        self.DCNN_mapping_2 = DCNN(
            input_channels=input_channels * 2,
            feature_extration_channels=feature_extration_channels,
            non_linear_channels=non_linear_channels,
            feature_extration_kernel_size=feature_extration_kernel_size,
            non_linear_mapping_kernel_size=non_linear_mapping_kernel_size,
            reconstruction_kernel_size=reconstruction_kernel_size,
            output_channels=output_channels,
        )

    def forward(
        self, coarse_img_01, coarse_img_02, coarse_img_03, fine_img_01, fine_img_03
    ):
        fine_img_diff_12 = self.DCNN_mapping_1(
            coarse_img_01, coarse_img_02, fine_img_01
        )  # fine_img_02 - fine_img_01
        fine_img_diff_23 = self.DCNN_mapping_2(
            coarse_img_02, coarse_img_03, fine_img_03
        )  # fine_img_03 - fine_img_02
        fine_img_diff_13 = (
            fine_img_diff_12 + fine_img_diff_23
        )  # fine_img_03 - fine_img_01
        return [fine_img_diff_12, fine_img_diff_23, fine_img_diff_13]


if __name__ == '__main__':
    model = STFNet()
    print(model)
