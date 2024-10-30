import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            self.optimizer.zero_grad()

        if self.is_train:
            with torch.no_grad():
                latent = self.vae.encode(batch["img"]).latent_dist.sample()
            # latent = batch["img"]
            all_losses = self.diffusion.get_loss(
                x_start=latent,
                classes=batch["label"],
            )
            batch["latent"] = latent
            batch.update(all_losses)

            # Backward pass and optimizer step
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            # DDIM sampling
            size = batch["img"].size(0)
            x_self_cond = batch.get("x_self_cond", None)
            classes = batch.get("label", None)
            eta = 0.0  # DDIM deterministic sampling

            # Generate samples with DDIM
            sampled_latents = self.diffusion.p_sample(
                size=(
                    size,
                    4,
                    8,
                ),
                x_self_cond=x_self_cond,
                classes=classes,
                eta=eta,
            )
            outputs = {"sampled_imgs": self.vae.decode(sampled_latents).sample}
            batch.update(outputs)

        # Update metrics
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        return batch
