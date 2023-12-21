import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderV1(nn.Module):

    def __init__(self, in_channels, hidden_dims) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, inputs):

        return self.encoder(inputs)


class DecoderV1(nn.Module):

    def __init__(self, in_channels, hidden_dims=None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=in_channels,
                      kernel_size=3,
                      padding=1), nn.Tanh())

    def forward(self, inputs):
        x = self.decoder(inputs)
        x = self.final_layer(x)
        return x


class AeV1(nn.Module):
    __doc__ = """ Vanilla VAE from  AntixK/PyTorch-VAE"""

    def __init__(self,
                 seed,
                 latent_dim: int = 128,
                 hidden_dims: list = None,
                 in_channels: int = 3,
                 kld_weight=1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False) -> None:
        super().__init__(seed=seed,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual)

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1), nn.Tanh())
