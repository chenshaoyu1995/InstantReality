#!/usr/bin/env python3
import sys
from subprocess import PIPE, Popen

import numpy as np
import visdom
from console_progressbar import ProgressBar

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer:
    def __init__(self, opt):
        self.vis = visdom.Visdom(
            server="http://localhost", port=8097, env=opt.env
        )
        print("Check out training progress at http://localhost:8097")
        if not self.vis.check_connection():
            self.create_visdom_connections()

        self.plot_data = {}
        self.pb = ProgressBar(total=opt.num_epochs)

    def create_visdom_connections(self):
        cmd = f"{sys.executable} -m visdom.server -p 8097 &>/dev/null &"
        print(
            "\n\nCould not connect to visdom server.\n"
            "Trying to start a server..."
        )
        print(f"Command: {cmd}")
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def print_progress_bar(self, epoch, counter_ratio):
        self.pb.print_progress_bar(epoch + counter_ratio)

    def plot_series(self, label, win_id, x, ys):
        if label not in self.plot_data:
            self.plot_data[label] = {
                "X": [],
                "Y": [],
                "title": label,
                "legend": list(ys.keys()),
                "ylabel": "loss",
            }
        self.plot_data[label]["X"].append(x)
        self.plot_data[label]["Y"].append(
            [ys[k] for k in self.plot_data[label]["legend"]]
        )
        self._plot(label, win_id)

    def plot_whole(self, label, win_id, X, Ys):
        self.plot_data[label] = {
            "X": X,
            "Y": [[Ys[k][idx] for k in Ys.keys()] for idx in range(len(X))],
            "title": label,
            "legend": list(Ys.keys()),
            "ylabel": "loss",
        }
        self._plot(label, win_id)

    def _plot(self, label, win_id):
        if label not in self.plot_data:
            raise Exception("Attribute %s does not exist", label)
        plot_data = self.plot_data[label]
        try:
            X, Y = self._fix_visdom_bug(plot_data)
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    "title": plot_data["title"],
                    "legend": plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": plot_data["ylabel"],
                },
                win=win_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connection()

    @staticmethod
    def _fix_visdom_bug(plot_data):
        if len(plot_data["legend"]) == 1:
            X = np.array(plot_data["X"])
            Y = np.array(plot_data["Y"]).flatten()
        else:
            X = np.stack(
                [np.array(plot_data["X"])] * len(plot_data["legend"]),
                1,
            )
            Y = np.array(plot_data["Y"])
        return X, Y
