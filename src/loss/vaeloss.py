import torch
import torch.nn.functional as F
from torch import nn


class VAELoss(nn.Module):
    def __init__(self, reconstruction_loss_type="mse"):
        super().__init__()
        if reconstruction_loss_type == "mse":
            self.reconstruction_loss = F.mse_loss
        elif reconstruction_loss_type == "bce":
            self.reconstruction_loss = F.binary_cross_entropy
        else:
            raise ValueError(
                f"Unknown reconstruction loss type: {reconstruction_loss_type}"
            )

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        **batch,
    ):
        batch_size = x.size(0)

        recon_loss = self.reconstruction_loss(
            recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction="sum"
        )

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + KLD

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": KLD}
