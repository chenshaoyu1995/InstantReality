#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import DATA_PATH, NUM_POPPING_VECTORS, get_net_path
from dataset import FoviatedLODDataset
from network import SaccadeNet
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O Config
    parser.add_argument("--datapath", type=str, default=DATA_PATH)
    parser.add_argument("--weightspath", type=str, default="./weights/")
    # Training Config
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    # GPU Config
    parser.add_argument("--disable_cuda", action="store_true")
    # Visualization Config
    parser.add_argument("--env", type=str, default="test_saccade")
    parser.add_argument("--triangle_id", type=int, default=0)
    opt = parser.parse_args()

    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    visualizer = Visualizer(opt)

    dataset = FoviatedLODDataset(opt.datapath, mode="test")
    dataloader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=1
    )

    net = SaccadeNet().to(device=opt.device)
    popping_criterions = [
        nn.L1Loss().to(device=opt.device) for _ in range(NUM_POPPING_VECTORS)
    ]

    net_path = get_net_path(opt, "saccade_net", opt.num_epochs)
    print(f"Loading weights in {net_path}")
    net.load_state_dict(torch.load(net_path, map_location=opt.device))

    with torch.no_grad():
        for data in dataloader:
            inps = data["input"]["camera"].to(opt.device)
            popping_labels = [
                data["output"]["no_mask_popping_density_list"][pi].to(
                    opt.device
                )
                for pi in range(NUM_POPPING_VECTORS)
            ]

            outputs = net(inps)

            area_mask = data["output"]["area"].clone().detach()
            area_mask[area_mask != 0.0] = 1.0

            loss = 0
            for pi in range(NUM_POPPING_VECTORS):
                loss += popping_criterions[pi](
                    outputs["no_mask_popping_density_list"][pi] * area_mask,
                    popping_labels[pi],
                )
            print(loss.item())

            sample_target_output = {
                "no_mask_popping_density_list": [
                    data["output"]["no_mask_popping_density_list"][pi][
                        :, opt.triangle_id
                    ]
                    .cpu()
                    .numpy()
                    for pi in range(NUM_POPPING_VECTORS)
                ],
            }
            area = data["output"]["area"].to(opt.device)
            sample_gen_output = {
                "no_mask_popping_density_list": [
                    (outputs["no_mask_popping_density_list"][pi])[
                        :, opt.triangle_id
                    ]
                    .cpu()
                    .numpy()
                    for pi in range(NUM_POPPING_VECTORS)
                ],
            }

    for pi in range(NUM_POPPING_VECTORS):
        visualizer.plot_whole(
            f"No Mask Popping Scores [{pi}] for seq9",
            pi,
            list(
                range(
                    len(sample_gen_output["no_mask_popping_density_list"][pi])
                )
            ),
            {
                "Target": sample_target_output["no_mask_popping_density_list"][
                    pi
                ],
                "Gen": sample_gen_output["no_mask_popping_density_list"][pi],
            },
        )
