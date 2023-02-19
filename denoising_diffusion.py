import torch
import torch.nn as nn
import torch.nn.functional as F
from img_utils import *
from tqdm import tqdm
# from Trainer import *


def expand_dims(x):
    """Expands to 3D tensor"""
    return x.reshape(-1, 1, 1).float()


class DenoisingDiffusion(nn.Module):
    def __init__(self, model, n_steps = 1000):
        super().__init__()
        self.register_module('model', model)
        self.register_buffer('n_steps', torch.tensor(n_steps))
        self.register_buffer('s', torch.Tensor([0.008]))
        self.register_buffer('alpha_bar',
                             self._f(torch.arange(n_steps + 1, dtype=torch.float64)) / self._f(torch.Tensor([0.])))
        self.register_buffer('alpha_bar_t_minus_1',
                             expand_dims(torch.concat((torch.Tensor([1.]), self.alpha_bar[:-1]))))
        self.register_buffer('alpha_bar', expand_dims(self.alpha_bar))
        self.register_buffer('beta',
                             expand_dims(torch.clamp(1. - (self.alpha_bar / self.alpha_bar_t_minus_1), max=0.999)))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar_sqrt', self.alpha_bar ** 0.5)
        self.register_buffer('one_minus_alpha_bar_sqrt', (1 - self.alpha_bar) ** 0.5)
        self.register_buffer('one_minus_alpha_bar_t_minus_one_sqrt', (1 - self.alpha_bar_t_minus_1) ** 0.5)

        self.register_buffer('beta_sqrt', self.beta ** 0.5)

        self.register_buffer('alpha_bar_reciprocal_sqrt', (1 / self.alpha_bar) ** 0.5)
        self.register_buffer('alpha_bar_minus_one_reciprocal_sqrt', (1 / self.alpha_bar - 1) ** 0.5)
        self.register_buffer('alpha_bar_t_minus_one_sqrt', self.alpha_bar_t_minus_1 ** 0.5)

        self.register_buffer('pos_var', self.beta * (1. - self.alpha_bar_t_minus_1) / (1. - self.alpha_bar))
        self.register_buffer('pos_var_sqrt', self.pos_var ** 0.5)

        self.register_buffer('x0_pos_coef', ((self.alpha_bar_t_minus_1 ** 0.5) * self.beta) / (1. - self.alpha_bar))
        self.register_buffer('xt_pos_coef',
                             ((self.alpha ** 0.5) * (1 - self.alpha_bar_t_minus_1)) / (1 - self.alpha_bar))

        self.register_buffer('eps_coef', self.beta / self.one_minus_alpha_bar_sqrt)
        self.register_buffer('alpha_reciprocal_sqrt', 1 / (self.alpha ** 0.5))

    def forward_process(self, x0, t):
        """Sample q(x0|xt) according to variance schedule"""
        eps = torch.randn_like(x0, device=self.device)
        mean = self.alpha_bar_sqrt[t] * x0
        var = self.one_minus_alpha_bar_sqrt[t]
        return mean + var * eps, eps

    @torch.no_grad()
    def x_0(self, xt, t, clipping='static', p=0.9):
        eps_theta = self.model(xt, t)
        x0 = self.alpha_bar_reciprocal_sqrt[t] * xt - self.alpha_bar_minus_one_reciprocal_sqrt[t] * eps_theta
        if clipping == "static":
            return x0.clamp(-1, 1)
        elif clipping == "dynamic":
            s = torch.quantile(abs(x0), p)
            s = torch.maximum(s, torch.Tensor([1], device=self.device))
            return x0.clamp(-s, s) / s
        elif clipping is None:
            return x0

    def reverse_process(self, xt, t):
        t = torch.tensor([t], device=self.device)
        x0 = self.x_0(xt, t)
        pos_mean = self.x0_pos_coef[t] * x0 + self.xt_pos_coef[t] * xt
        eps = torch.randn_like(xt)
        return pos_mean + self.pos_var_sqrt[t] * eps

    @torch.no_grad()
    def alternative_reverse(self, xt, t):
        t = torch.tensor([t], device=self.device)
        eps_theta = self.model(xt, t)
        eps_theta *= self.eps_coef[t]
        mu = self.alpha_reciprocal_sqrt[t] * (xt - eps_theta)
        eps = torch.randn_like(xt, device=self.device)
        return mu + self.pos_var_sqrt[t] * eps

    def DDIM(self, xt, t, alpha_bar_t_minus_tau):
        t = torch.tensor([t], device=self.device)
        x0 = self.x_0(xt, t, clipping='static')
        return (alpha_bar_t_minus_tau ** 0.5) * x0 + (xt - self.alpha_bar_sqrt[t] * x0) / \
               self.one_minus_alpha_bar_sqrt[t] * ((1 - alpha_bar_t_minus_tau) ** 0.5)

    def simple_loss(self, x0):
        t = torch.randint(0, self.n_steps + 1, (x0.shape[0],), device=self.device)
        xt, eps = self.forward_process(x0, t)
        eps_theta = self.model(xt, t)
        return F.mse_loss(eps, eps_theta)

    @property
    def device(self):
        return self.n_steps.device

    def _f(self, t):
        return torch.cos(
            (((t / self.n_steps) + self.s) * torch.pi) / ((1. + self.s) * 2)
        ) ** 2

    def sample(self,
               n_samples,
               sampler="DDPM",
               denoising_steps=1000,
               display_frames=10,

               generate_unconditional=False,

               generate_variations=False,
               img_variability=0,
               reference_img_path=None,

               edits=False,
               mask_img_path=None,

               masked_edits=False,
               masked_edit_img_path=None,

               generate_gif=False,
               gif_frames=60,
               render = True

               # Used in
               ):
        xt = torch.randn(n_samples, 256, 64, device=self.device)
        gif_imgs = []
        variability = img_variability
        tau_step_diff = 1000 // denoising_steps
        steps = int(variability * denoising_steps * tau_step_diff)
        assert generate_variations + generate_unconditional + edits + masked_edits <= 1, "Only can use one type of generation!"
        if sampler == "DDPM":
            assert denoising_steps == 1000, "DDPM sampler only supports 1000 steps"
        elif sampler == "DDIM":
            assert (divmod(1000, denoising_steps)[1] == 0), "steps must be a factor of 1000 for DDIM"
            alpha_bar_t_minus_tau = torch.concat(
                (expand_dims(torch.Tensor([1.] * tau_step_diff).to(device)), self.alpha_bar[:-tau_step_diff])).to(
                self.device)
        else:
            raise AssertionError("Sampler must be DDPM or DDIM")
        if generate_variations or edits or masked_edits:
            org_xt = tensor_from_path(reference_img_path, device, n_samples)
            if masked_edits:
                assert mask_img_path, "Mask needed, if you don't want to edit only a part of the img with a base " \
                                      "starting img use generate_variations "
                xt = self.forward_process(tensor_from_path(masked_edit_img_path, device, n_samples), steps)[0]
            else:
                xt = self.forward_process(org_xt, steps)[0]
            if edits or masked_edits:
                reference_img = img_from_path(reference_img_path, device)
                mask_img = img_from_path(mask_img_path, device)
                mask = flatten(get_mask(reference_img, mask_img)).repeat(n_samples, 1, 1)


        gif_every = denoising_steps // gif_frames
        show_every = denoising_steps // display_frames

        for t in tqdm(range(steps, 0, - tau_step_diff), unit="steps"):
            xt = self.DDIM(xt, t, alpha_bar_t_minus_tau[t]) if sampler == "DDIM" else self.reverse_process(xt, t)
            if edits or masked_edits:
                org_xt_noised = self.forward_process(org_xt, t)[0]
                xt = get_masked_img(mask, org_xt_noised, xt)
            true_steps = t / tau_step_diff
            g = display_img_from_tensor(xt, display=true_steps % show_every == 0,
                                        gif=generate_gif and true_steps % gif_every == 0,render = render,binary_alpha=False)
            if g is not None:
                gif_imgs.append(g)
        save_img_to("Samples", xt,binary_alpha=False)
        if generate_gif:
            img_to_gif(gif_imgs, "Samples.gif")
