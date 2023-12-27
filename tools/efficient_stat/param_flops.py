import torch
from thop import profile
from src.model import (
    SwinSTFM,
    OPGANGenerator,
    OPGANDiscriminator,
    MSDiscriminator,
    SFFusion,
    STFDCNN,
    STFGANDiscriminator,
    STFGANGenerator,
    GaussianDiffusion,
    PredNoiseNet,
    BiGaussianDiffusion,
    BiPredNoiseNet,
)

M1 = torch.randn(1, 6, 256, 256)
M2 = torch.randn(1, 6, 256, 256)
M3 = torch.randn(1, 6, 256, 256)
L1 = torch.randn(1, 6, 256, 256)
L2 = torch.randn(1, 6, 256, 256)
L3 = torch.randn(1, 6, 256, 256)

model = SwinSTFM()
flops, params = profile(
    model,
    inputs=(
        M1,
        M2,
        L1,
    ),
)
print('SwinSTFM')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')


print('OPGAN')
model = OPGANGenerator()
flops, params = profile(
    model,
    inputs=(
        M1,
        M2,
        L1,
    ),
)
print('OPGANGenerator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

model = OPGANDiscriminator()
flops, params = profile(
    model,
    inputs=(L1,),
)
print('OPGANDiscriminator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

print('GANSTFM')
model = SFFusion()
flops, params = profile(
    model,
    inputs=((M1, L1),),
)
print('GANSTFMGenerator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

model = MSDiscriminator()
flops, params = profile(
    model,
    inputs=(torch.cat((L1, L2), 1),),
)
print('GANSTFMDiscriminator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')


print('STFDCNN')
model = STFDCNN(6)
flops, params = profile(
    model,
    inputs=(M1,),
)
print('STFDCNN')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

M = torch.randn(1, 6, 64, 64)
print('STFGAN')
model = STFGANGenerator(6)
flops, params = profile(
    model,
    inputs=(
        M,
        M,
        M,
        L3,
        L1,
    ),
)
print('STFGANGenerator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

model = STFGANDiscriminator(6)
flops, params = profile(
    model,
    inputs=(L1,),
)
print('STFGANDiscriminator')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')

print('GaussianDiffusion')
model = GaussianDiffusion(
    model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    image_size=256,
    timesteps=100,
    sampling_timesteps=50,
    objective="pred_x0",
    ddim_sampling_eta=0.0,
)
flops, params = profile(
    model,
    inputs=(
        M1,
        M2,
        L1,
        L2,
    ),
)
print('GaussianDiffusion')
print('FLOPs = ' + str(flops / 1000**3) + 'G')
print('Params = ' + str(params / 1000**2) + 'M')
