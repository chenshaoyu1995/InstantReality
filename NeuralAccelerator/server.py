#!/usr/bin/env python3
import argparse

import torch

from constants import NUM_POPPING_VECTORS, get_net_path
from network import FixationNet, SaccadeNet

import zmq
import json
import numpy as np

ZMQ_PORT = 5555

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weightspath", type=str, default="./weights/")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--disable_cuda", action="store_true")

    opt = parser.parse_args()

    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device("cpu")

    net_fix = FixationNet().to(device=opt.device)
    net_path = get_net_path(opt, "fixation_net", opt.num_epochs)
    print(f"Loading weights in {net_path}")
    net_fix.load_state_dict(torch.load(net_path, map_location=opt.device))
    
    net_sac = SaccadeNet().to(device=opt.device)
    net_path = get_net_path(opt, "saccade_net", opt.num_epochs)
    print(f"Loading weights in {net_path}")
    net_sac.load_state_dict(torch.load(net_path, map_location=opt.device))

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:{}".format(ZMQ_PORT))  
    print('Server ready')
        
    with torch.no_grad():
        while True:
            received = json.loads(socket.recv().decode())
            respond = {}    

            if received["isSaccade"] == 1:
                inp = np.concatenate(
                    (
                        np.array(received["eye"], dtype=np.float32),
                        np.array(received["lookat"], dtype=np.float32),
                        np.array(received["up"], dtype=np.float32),
                    )
                )
                inp = torch.from_numpy(inp[np.newaxis,:]).to(opt.device)
                output = net_sac(inp)
                for pi in range(NUM_POPPING_VECTORS):
                   respond[str(pi)] = output["no_mask_popping_density_list"][pi].squeeze().tolist() 
            else:
                inp = np.concatenate(
                    (
                        np.array(received["eye"], dtype=np.float32),
                        np.array(received["lookat"], dtype=np.float32),
                        np.array(received["up"], dtype=np.float32),
                        np.array(received["gaze"], dtype=np.float32),
                    )
                )
                inp = torch.from_numpy(inp[np.newaxis,:]).to(opt.device)
                output = net_fix(inp)
                for pi in range(NUM_POPPING_VECTORS):
                   respond[str(pi)] = output["popping_density_list"][pi].squeeze().tolist() 

            socket.send(json.dumps(respond).encode('utf-8'))
