import torch
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from src.metrics.base_metric import BaseMetric


class FIDMetric(BaseMetric):
    def __init__(self, name=None, device="cuda", dims=2048):
        super().__init__(name=name)
        self.device = device
        self.dims = dims
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(self.device)
        self.inception.eval()

    @torch.no_grad()
    def _get_inception_features(self, images):
        if images.shape[-1] != 299:
            images = F.interpolate(
                images,
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
        features = self.inception(images)[0]
        return features

    def __call__(self, img: torch.Tensor, sampled_imgs: torch.Tensor, **kwargs):
        real_images, fake_images = (
            img,
            sampled_imgs,
        )

        real_features = self._get_inception_features(real_images).cpu().numpy()
        fake_features = self._get_inception_features(fake_images).cpu().numpy()

        fid_score = calculate_frechet_distance(real_features, fake_features)

        return torch.tensor(fid_score)
