import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(a, t, x_shape):
    # Taken from Annotated diffusion model by HuggingFace
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_params(betas, timesteps):
    # Adopted from Annotated diffusion model by HuggingFace
    # define alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return (
        betas,
        sqrt_recip_alphas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance,
    )


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(t.device), beta.to(t.device)], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(model_args, seq, model, b, eta):
    with torch.no_grad():
        x = model_args[0]
        self_cond = model_args[1]
        clas_lbls = model_args[2]
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        progress_bar = tqdm(
            zip(reversed(seq), reversed(seq_next)),
            desc="DDIM Sampling",
            total=len(seq),
            mininterval=0.5,
            leave=False,
            disable=False,
            colour="#F39C12",
            dynamic_ncols=True,
        )

        # for i, j in zip(reversed(seq), reversed(seq_next)):
        for i, j in progress_bar:
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, x_self_cond=self_cond, lbls=clas_lbls).detach()
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.detach().to("cpu"))

    return xs, x0_preds


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps: int = 500,
        sample_every: int = 5,
        device: str = "cuda",
    ):
        super(Diffusion, self).__init__()
        self.timesteps = timesteps
        self.sample_every = sample_every
        self.device = device
        # betas = cosine_beta_schedule(timesteps=timesteps)
        betas = torch.linspace(0.0001, 0.02, timesteps)
        dif_params = forward_diffusion_params(betas, timesteps)
        self.betas = dif_params[0]
        self.sqrt_recip_alphas = dif_params[1]
        self.sqrt_alphas_cumprod = dif_params[2]
        self.sqrt_one_minus_alphas_cumprod = dif_params[3]
        self.posterior_variance = dif_params[4]
        self.loss = nn.MSELoss()
        self.model = model

    def forward(self, x, t, x_self_cond=None, classes=None):
        """Forward pass through the U-Net model to predict noise"""
        return self.model(x, t, x_self_cond=x_self_cond, lbls=classes)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_loss(self, x_start, t=None, noise=None, x_self_cond=None, classes=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            t = torch.randint(
                0, self.timesteps, size=(x_start.size(0),), device=self.device
            )
        # Generate the noisy image using q_sample
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict noise with the model
        predicted_noise = self.forward(
            x_noisy, t, x_self_cond=x_self_cond, classes=classes
        )

        # Calculate the loss
        return {"loss": self.loss(noise, predicted_noise)}

    def p_sample(self, size, x_self_cond=None, classes=None, eta=1.0):
        x = torch.randn(*size, device=self.device)
        seq = range(0, self.timesteps, self.sample_every)
        seq = [int(s) for s in list(seq)]
        model_args = (x, x_self_cond, classes)
        xs = generalized_steps(model_args, seq, self.model, self.betas, eta=eta)
        return xs[0][-1]
