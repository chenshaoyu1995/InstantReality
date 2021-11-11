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
    parser.add_argument("--num_epochs", type=int, default=200)
    # GPU Config
    parser.add_argument("--disable_cuda", action="store_true")
    # Visualization Config
    parser.add_argument("--env", type=str, default="saccade")
    opt = parser.parse_args()

    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    visualizer = Visualizer(opt)

    train = FoviatedLODDataset(opt.datapath, mode="train")
    train_loader = DataLoader(
        train, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    val = FoviatedLODDataset(opt.datapath, mode="validation")
    val_loader = DataLoader(
        val, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )
    test = FoviatedLODDataset(opt.datapath, mode="test")
    test_loader = DataLoader(
        test, batch_size=opt.batch_size, shuffle=True, num_workers=1
    )

    net = SaccadeNet().to(device=opt.device)
    popping_criterions = [
        nn.L1Loss().to(device=opt.device) for _ in range(NUM_POPPING_VECTORS)
    ]
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

    total_iters = 0
    for epoch in range(1, opt.num_epochs + 1):
        epoch_iters = 0

        # Validation & Test
        with torch.no_grad():
            val_total = 0
            val_loss = 0
            for i, data in enumerate(val_loader):
                inps = data["input"]["camera"].to(opt.device)
                popping_labels = [
                    data["output"]["no_mask_popping_density_list"][pi].to(
                        opt.device
                    )
                    for pi in range(NUM_POPPING_VECTORS)
                ]

                outputs = net(inps)

                area_mask = data["output"]["area"].to(opt.device)
                area_mask[area_mask != 0.0] = 1.0

                loss = 0
                for pi in range(NUM_POPPING_VECTORS):
                    loss += popping_criterions[pi](
                        outputs["no_mask_popping_density_list"][pi]
                        * area_mask,
                        popping_labels[pi],
                    )
                val_total += 1
                val_loss += loss.item()

            visualizer.plot_series(
                "validation_losses",
                2,
                epoch,
                {"Loss": val_loss / val_total},
            )

            test_total = 0
            test_loss = 0
            for i, data in enumerate(test_loader):
                inps = data["input"]["camera"].to(opt.device)
                popping_labels = [
                    data["output"]["no_mask_popping_density_list"][pi].to(
                        opt.device
                    )
                    for pi in range(NUM_POPPING_VECTORS)
                ]

                outputs = net(inps)

                area_mask = data["output"]["area"].to(opt.device)
                area_mask[area_mask != 0.0] = 1.0

                loss = 0
                for pi in range(NUM_POPPING_VECTORS):
                    loss += popping_criterions[pi](
                        outputs["no_mask_popping_density_list"][pi]
                        * area_mask,
                        popping_labels[pi],
                    )
                test_total += 1
                test_loss += loss.item()

            visualizer.plot_series(
                "test_losses",
                1,
                epoch,
                {"Loss": test_loss / test_total},
            )

        # Train
        for i, data in enumerate(train_loader):
            total_iters += opt.batch_size
            epoch_iters += opt.batch_size

            inps = data["input"]["camera"].to(opt.device)
            popping_labels = [
                data["output"]["no_mask_popping_density_list"][pi].to(
                    opt.device
                )
                for pi in range(NUM_POPPING_VECTORS)
            ]

            optimizer.zero_grad()
            outputs = net(inps)

            area_mask = data["output"]["area"].to(opt.device)
            area_mask[area_mask != 0.0] = 1.0

            loss = 0
            for pi in range(NUM_POPPING_VECTORS):
                loss += popping_criterions[pi](
                    outputs["no_mask_popping_density_list"][pi] * area_mask,
                    popping_labels[pi],
                )
            loss.backward()
            optimizer.step()

            if total_iters % 128 == 0:
                visualizer.print_progress_bar(
                    epoch - 1,
                    float(epoch_iters) / len(train),
                )
                visualizer.plot_series(
                    "training_loss",
                    0,
                    epoch + float(epoch_iters) / len(train),
                    {"Loss": loss.item()},
                )

        if epoch % 100 == 0:
            net_path = get_net_path(opt, "saccade_net", epoch)
            torch.save(net.state_dict(), net_path)
            print("Saved net at %s" % net_path)
