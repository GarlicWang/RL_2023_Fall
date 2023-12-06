import torch
from torch import nn


class DomainTranslater(nn.Module):
    @staticmethod
    def basic_block(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Module(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
        )

    def __init__(self, input_dim: int = 1024, output_dim: int = 1024, latent_dim: int = 256) -> None:
        super().__init__()

        self.encoder = nn.Module(
            DomainTranslater.basic_block(input_dim, latent_dim * 8),
            DomainTranslater.basic_block(latent_dim * 8, latent_dim * 4),
            DomainTranslater.basic_block(latent_dim * 4, latent_dim * 2),
            DomainTranslater.basic_block(latent_dim * 2, latent_dim),
        )
        self.decoder = nn.Module(
            DomainTranslater.basic_block(latent_dim, latent_dim * 2),
            DomainTranslater.basic_block(latent_dim * 2, latent_dim * 4),
            DomainTranslater.basic_block(latent_dim * 4, latent_dim * 8),
            DomainTranslater.basic_block(latent_dim * 8, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)
