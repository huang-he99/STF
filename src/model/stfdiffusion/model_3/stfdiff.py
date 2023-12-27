import math
from functools import partial
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm


# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.0,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond

        self.denoiser = model
        self.channels = self.denoiser.channels
        self.self_condition = self.denoiser.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise',
            'pred_x0',
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        register_buffer(
            'p2_loss_weight',
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        combined_fine_img_t,
        t,
        x_self_cond=None,
        clip_x_start=False,
    ):
        model_output = self.denoiser(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            combined_fine_img_t,
            t,
            x_self_cond,
        )
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(combined_fine_img_t, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(combined_fine_img_t, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        combined_fine_img_t,
        t,
        x_self_cond=None,
        clip_denoised=True,
    ):
        preds = self.model_predictions(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            combined_fine_img_t,
            t,
            x_self_cond,
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        x_0_bar = x_start
        x_t_direct = (
            extract(self.sqrt_alphas_cumprod, t, x_0_bar.shape) * x_0_bar
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0_bar.shape)
            * fine_img_01
        )

        t_add1 = t + 1
        if t_add1.data == 1000:
            x_t_add1_bar = fine_img_01
        else:
            x_t_add1_bar = (
                extract(self.sqrt_alphas_cumprod, t_add1, x_0_bar.shape) * x_0_bar
                + extract(self.sqrt_one_minus_alphas_cumprod, t_add1, x_0_bar.shape)
                * fine_img_01
            )
        x_t_res = combined_fine_img_t - x_t_add1_bar + x_t_direct
        return x_t_res, x_0_bar

    @torch.no_grad()
    def p_sample(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        combined_fine_img_t,
        t: int,
        x_self_cond=None,
        clip_denoised=True,
    ):
        b, *_, device = *combined_fine_img_t.shape, combined_fine_img_t.device
        batched_times = torch.full(
            (combined_fine_img_t.shape[0],),
            t,
            device=combined_fine_img_t.device,
            dtype=torch.long,
        )
        x_t_res, x_0_bar = self.p_mean_variance(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            combined_fine_img_t,
            t=batched_times,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )
        return x_t_res, x_0_bar

    @torch.no_grad()
    def p_sample_loop(
        self, coarse_img_01, coarse_img_02, fine_img_01, combined_fine_img_t
    ):
        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            combined_fine_img_t, x_start = self.p_sample(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                combined_fine_img_t,
                t,
                self_cond,
            )

        # img = unnormalize_to_zero_to_one(img)
        return combined_fine_img_t

    @torch.no_grad()
    def ddim_sample(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        clip_denoised=True,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            noisy_fine_img_02.shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.denoiser_predictions(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                time_cond,
                self_cond,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                noisy_fine_img_02 = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(noisy_fine_img_02)

            noisy_fine_img_02 = (
                x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            )

        return noisy_fine_img_02

    @torch.no_grad()
    def sample(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
    ):
        combined_fine_img_t = fine_img_01.clone()
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            combined_fine_img_t,
        )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
            reversed(range(0, t)), desc='interpolation sample time step', total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *coarse_img_01.shape,
            coarse_img_01.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # noise = torch.randn_like(fine_img_02)
        combined_fine_img_t = self.diffusion_process(
            x_0=fine_img_02, t=t, noise=fine_img_01
        )

        denoiser_output = self.denoiser(
            coarse_img_01, coarse_img_02, fine_img_01, combined_fine_img_t, t
        )

        if self.objective == 'pred_noise':
            target = combined_fine_img_t
        elif self.objective == 'pred_x0':
            target = fine_img_02
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(denoiser_output, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def diffusion_process(self, x_0, t, noise):
        noisy_x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        return noisy_x_t
