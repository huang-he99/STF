from torch import nn


class STFDCNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        feature_extration_channels=64,
        non_linear_channels=32,
        feature_extration_kernel_size=9,
        non_linear_mapping_kernel_size=5,
        reconstruction_kernel_size=5,
        out_channels=1,
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
                out_channels=out_channels,
                kernel_size=reconstruction_kernel_size,
                padding="same",
            )
        )

    def forward(self, coarse_input):
        fine_coarse_residual = self.reconstruction_layer(
            self.nonlinear_mapping_layer(self.feature_extraction_layer(coarse_input))
        )
        fine_output = coarse_input + fine_coarse_residual
        return fine_output


if __name__ == '__main__':
    model = STFDCNN()
    print(model)
