import random
import torch
import os
import logging
import numpy as np
import tqdm
from gpt_models.autoregressive_wrapper import AutoregressiveWrapper
from torch.utils.data import DataLoader
from torchviz import make_dot, make_dot_from_trace

from utils import parse_args
from train import decode_tokens, train_dataset, GENERATE_LENGTH, cycle
from IPython.display import Image

args = parse_args()

it = 0
inps = []
while it<100:
    inp_loader = cycle(DataLoader(train_dataset))
    inps.append(next(inp_loader))
    it += 1
print("loaded inps")

losses = []
for i in tqdm.tqdm(range(1000), desc='testing'):
  epoch = i * 100

with open('wikitext2_test2.txt', 'w') as f:
    inp = random.choice(train_dataset)[:-1]
    prime = decode_tokens(inp)
    print(args.dataset,file=f)
    print("input:",file=f)
    print(prime,file=f)


    for epoch in [100, 500,1000, 5000, 10000, 20000, 30000, 50000, 99900]:
        print(epoch/100)
        for model_name in ["gpt_gmlp", "gpt_rs", "gpt_sa"]:
            model = torch.load(f'./{args.results_dir}/{model_name}_{args.dataset}_ep_{epoch}.pt').to(args.gpu).cuda()
            model = AutoregressiveWrapper(model)
            model.eval()
            sample = model.generate(inp.to(0), GENERATE_LENGTH)

        
            
            output_str = decode_tokens(sample)
            string_epoch = epoch /100
            print(f"{model_name}_epoch_{string_epoch} output:", file=f)
            print(output_str,file=f)

# inp = random.choice(val_dataset)[:-1]
# prime = decode_tokens(inp)
# # print(args.dataset,file=f)
# # print("input:")
# # print(prime)

# for epoch in [100]:
#         print(epoch/100)
#         for model_name in ["gpt_gmlp", "gpt_rs", "gpt_sa"]:
#             model = torch.load(f'./{args.results_dir}/{model_name}_{args.dataset}_ep_{epoch}.pt').to(args.gpu).cuda()
            
#             model = AutoregressiveWrapper(model)

#             model.eval()
#             sample = model.generate(inp.to(0), GENERATE_LENGTH)
#             sample = sample.float()
#             # sample = model(inp.to(0))

#             dot = make_dot(sample,params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
#             print(model_name)
#             print(dot)
            # dot.save(f'{model_name}_graph.png')  # Save the graph to a file
            # dot.render()
            
            # output_str = decode_tokens(sample)
            # string_epoch = epoch /100
            # print(f"{model_name}_epoch_{string_epoch} output:", file=f)
            # print(output_str,file=f)






#   loss = model.bcp_loss(inps, args.gpu)
#   losses.append(loss)

# avg_loss = np.mean(losses)


# save losses
# np.save(f'./{args.results_dir}/losses/{args.model}_{args.dataset}_losses.npy', np.array(losses))
