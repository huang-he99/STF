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
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

M1 = torch.randn(1, 6, 256, 256).cuda()
M2 = torch.randn(1, 6, 256, 256).cuda()
M3 = torch.randn(1, 6, 256, 256).cuda()
L1 = torch.randn(1, 6, 256, 256).cuda()
L2 = torch.randn(1, 6, 256, 256).cuda()
L3 = torch.randn(1, 6, 256, 256).cuda()

M = torch.randn(1, 6, 64,64).cuda()

M1_patch = torch.randn(1, 6, 256, 256).cuda()
M2_patch = torch.randn(1, 6, 256, 256).cuda()
M3_patch = torch.randn(1, 6, 256, 256).cuda()
L1_patch = torch.randn(1, 6, 256, 256).cuda()
L2_patch = torch.randn(1, 6, 256, 256).cuda()
L3_patch = torch.randn(1, 6, 256, 256).cuda()


model = SwinSTFM().cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        # for _ in range(35):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        model(M1_patch, M1_patch, M1_patch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        t = t + elapsed

print('yure')
print('Time = ' + str(t/4) + 's')

del model
torch.cuda.empty_cache()

model = SwinSTFM().cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        # for _ in range(35):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        model(M1_patch, M2_patch, L1_patch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        t = t + elapsed

print('SwinSTFM')
print('Time = ' + str(t/4) + 's')

del model
torch.cuda.empty_cache()

model = OPGANGenerator().cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        model(M1, M2, L1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        t = t + elapsed

print('OPGAN')
print('Time = ' + str(t/4) + 's')

del model
torch.cuda.empty_cache()
model = SFFusion().cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        torch.cuda.synchronize()
        start_time = time.time()
        model((M1, L1))
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        t = t + elapsed

print('GANSTFMGenerator')
print('Time = ' + str(t/4) + 's')

del model
torch.cuda.empty_cache()
model = STFDCNN(6).cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        torch.cuda.synchronize()
        start_time = time.time()
        model(M1)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        t = t + elapsed

print('STFDCNN')
print('Time = ' + str(t/2) + 's')

del model
torch.cuda.empty_cache()
model = STFGANGenerator(6).cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        torch.cuda.synchronize()
        start_time = time.time()
        model(M,M,M,L1,L1)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        t = t + elapsed

print('STFGANGenerator')
print('Time = ' + str(t/2) + 's')

del model
torch.cuda.empty_cache()
# print('GaussianDiffusion')
model = GaussianDiffusion(
    model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    image_size=256,
    timesteps=100,
    sampling_timesteps=10,
    objective="pred_x0",
    ddim_sampling_eta=0.0,
).cuda()
with torch.no_grad():
    t = 0 
    for _ in range(4):
        torch.cuda.synchronize()
        start_time = time.time()
        model.sample(M1,M2,L2)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        t = t + elapsed

print('GaussianDiffusion')
print('Time = ' + str(t/4) + 's')

