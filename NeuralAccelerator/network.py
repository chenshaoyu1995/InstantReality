#!/usr/bin/env python3
import torch.nn as nn

from constants import (
    CAMERA_VECTOR_DIM,
    GAZE_VECTOR_DIM,
    NUM_POPPING_VECTORS,
    NUM_TRIANGLES,
)


class ResidualBlock(nn.Module):
    def __init__(self, inp_dim):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, inp_dim),
            nn.ReLU(),
            nn.Linear(inp_dim, inp_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, inp):
        return self.activation(self.model(inp) + inp)


def buildMLP(inp_dim, out_dim, layer_sizes):
    layers = [nn.Linear(inp_dim, layer_sizes[0]), nn.ReLU()]
    for idx in range(len(layer_sizes) - 1):
        layers.extend(
            [
                ResidualBlock(layer_sizes[idx]),
                nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]),
                nn.ReLU(),
            ]
        )
    layers.extend(
        [
            nn.Linear(layer_sizes[-1], out_dim),
            nn.Sigmoid(),
        ]
    )
    return nn.Sequential(*layers)


class FixationNet(nn.Module):
    def __init__(self):
        super(FixationNet, self).__init__()
        self.popping_models = nn.ModuleList(
            [
                buildMLP(
                    CAMERA_VECTOR_DIM + GAZE_VECTOR_DIM,
                    NUM_TRIANGLES,
                    [100, 1000, 1000],
                )
                for _ in range(NUM_POPPING_VECTORS)
            ]
        )
        # self.eccentricity_model = buildMLP(
        #     CAMERA_VECTOR_DIM + GAZE_VECTOR_DIM,
        #     NUM_TRIANGLES,
        #     [100, 1000, 1000],
        # )

    def forward(self, inp):
        return {
            "popping_density_list": [
                self.popping_models[pi](inp)
                for pi in range(NUM_POPPING_VECTORS)
            ],
            # "eccentricity_density": self.eccentricity_model(inp),
        }


class SaccadeNet(nn.Module):
    def __init__(self):
        super(SaccadeNet, self).__init__()
        self.popping_models = nn.ModuleList(
            [
                buildMLP(CAMERA_VECTOR_DIM, NUM_TRIANGLES, [100, 1000, 1000])
                for _ in range(NUM_POPPING_VECTORS)
            ]
        )

    def forward(self, inp):
        return {
            "no_mask_popping_density_list": [
                self.popping_models[pi](inp)
                for pi in range(NUM_POPPING_VECTORS)
            ],
        }
