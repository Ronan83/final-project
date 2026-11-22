"""Neural SDF model: positional encoding followed by a small MLP."""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Maps x to [x, sin(2^0 x), cos(2^0 x), ..., sin(2^(F-1) x), cos(2^(F-1) x)].

  The feature order is part of the export format and is mirrored in WGSL.
  """

    def __init__(self, num_frequencies: int) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer('frequencies',
                             2.0**torch.arange(num_frequencies, dtype=torch.float32),
                             persistent=False)

    @property
    def output_dim(self) -> int:
        return 3 * (1 + 2 * self.num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for frequency in self.frequencies:
            features += [torch.sin(frequency * x), torch.cos(frequency * x)]
        return torch.cat(features, dim=-1)


# Smooth activations give smooth SDFs; ReLU makes the surface piecewise planar,
# which shows up as faceted shading because normals come from SDF gradients.
ACTIVATIONS: dict[str, type[nn.Module]] = {'relu': nn.ReLU, 'silu': nn.SiLU}


class NeuralSdf(nn.Module):
    """Positional encoding + MLP with `hidden_layers` hidden layers of `hidden_width`."""

    def __init__(self,
                 num_frequencies: int,
                 hidden_width: int,
                 hidden_layers: int,
                 activation: str = 'silu') -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f'unknown activation {activation!r}; expected one of {list(ACTIVATIONS)}')
        self.activation = activation
        self.encoding = PositionalEncoding(num_frequencies)
        layers: list[nn.Module] = []
        in_features = self.encoding.output_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_features, hidden_width), ACTIVATIONS[activation]()]
            in_features = hidden_width
        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)

    def linear_layers(self) -> list[nn.Linear]:
        return [m for m in self.mlp if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(x)).squeeze(-1)
