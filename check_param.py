import random
import torch
import os
import logging
import numpy as np
import tqdm
from gpt_models.autoregressive_wrapper import AutoregressiveWrapper
from torch.utils.data import DataLoader
# from torchviz import make_dot, make_dot_from_trace

from utils import parse_args

args = parse_args()

model = torch.load(f'./{args.results_dir}/{args.model}_enwik8_ep_100.pt')

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)