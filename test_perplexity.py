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
from train import decode_tokens, val_dataset, GENERATE_LENGTH, cycle

args = parse_args()

# load the model
# model = torch.load(f'./{args.results_dir}/{args.model}_{args.dataset}_ep_{args.epochs}.pt').to(args.gpu)

# model = AutoregressiveWrapper(model)

# inference
print('Enter a string to generate from:')
# inp = random.choice(val_dataset)[:-1]
# prime = decode_tokens(inp)
# print(prime)
# # print(f'%s \n\n %s', (prime, '*' * 100))

# print('*' * 100)

# sample = model.generate(inp, GENERATE_LENGTH)
# output_str = decode_tokens(sample)
# print(output_str)
it = 0
inps = []
while it<10:
    inp_loader = cycle(DataLoader(val_dataset))
    inps.append(next(inp_loader))
    it += 1
print("loaded inps")

for model_name in ["gpt_gmlp", "gpt_rs", "gpt_sa"]:
  losses = []
  print(model_name)
  for i in tqdm.tqdm(range(1000), desc='testing'):
    epoch = i * 100
    model = torch.load(f'./{args.results_dir}/{model_name}_{args.dataset}_ep_{epoch}.pt').to(args.gpu).cuda()
    

    model = AutoregressiveWrapper(model)
    loss = model.perplexity(inps, args.gpu)
    print(loss)
    losses.append(loss)

  # avg_loss = np.mean(losses)


  # save losses
  np.save(f'./{args.results_dir}/losses/{model_name}_{args.dataset}_losses_pp.npy', np.array(losses))
