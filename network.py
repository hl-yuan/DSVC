import torch.nn as nn
from torch.nn.functional import normalize
import torch

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self, input_dim, feature_dim, activation='relu', batchnorm=True):
        super(Autoencoder, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm

        # Encoder definition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # Decoder definition
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            z: Latent representation of shape [batch_size, feature_dim].
            xr: Reconstructed input of shape [batch_size, input_dim].
        """
        z = self.encoder(x)
        xr = self.decoder(z)
        return z, xr


class DimReductionMLP(nn.Module):
    def __init__(self, input_dim=100,  output_dim=10):
        super(DimReductionMLP, self).__init__()

        # 使用 MLP 网络进行降维
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), int(input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(input_dim/4), int(input_dim / 8)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 8), output_dim),
        )

        self.output_layer = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x1 = normalize(x, dim=1)
        z = self.extractor(x1)

        return z























# class Autoencoder(nn.Module):
#     """AutoEncoder module that projects features to latent space."""
#     def __init__(self, activation='relu', batchnorm=True):
#         super(Autoencoder, self).__init__()
#         # self.input_dim = input_dim
#         # self.feature_dim = feature_dim
#         self._activation = activation
#         self._batchnorm = batchnorm
#
#     def Encoder(self, input_dim, feature_dim):
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(512, feature_dim),
#         )
#     def Decoder(self, input_dim, feature_dim):
#         self.decoder = nn.Sequential(
#             nn.Linear(feature_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, input_dim)
#         )
#     def forward(self, xs,  input_dim,  feature_dim):
#
#         x = xs
#         z = self.Encoder(x, input_dim, feature_dim)
#
#         xr = self.Decoder(z, input_dim, feature_dim)
#
#
#         return z, xr
