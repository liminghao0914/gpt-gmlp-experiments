from gpt_models.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import logging
from torchtext.data import Field, Iterator
from torchtext import datasets

from utils import parse_args
import gpt_models
# constants

args = parse_args()

NUM_BATCHES = args.epochs
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = args.lr
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 768
SEQ_LEN = 768
DEVICE = args.gpu

# helpers


def cycle(loader):
  while True:
    for data in loader:
      yield data


def decode_token(token):
  return str(chr(max(32, token)))


def decode_tokens(tokens):
  return ''.join(list(map(decode_token, tokens)))


model_options = {
    'num_tokens': 256,
    'dim': 512,
    'seq_len': SEQ_LEN,
    'depth': 4,
    'window': (16, 32, 64, SEQ_LEN),
}

model = gpt_models.__dict__[args.model](model_options).to(DEVICE)
model = AutoregressiveWrapper(model, device=DEVICE)
model.cuda().to(DEVICE)

# prepare enwik8 data

if True:
  TEXT = Field(lower=True, tokenize='spacy',tokenizer_language = 'en_core_web_sm', batch_first=True)
  train, dev, test = datasets.WikiText2.splits(TEXT)
  trX = np.fromstring(' '.join(train[0].text), dtype=np.uint8)
  vaX = np.fromstring(' '.join(dev[0].text), dtype=np.uint8)
  testX = np.fromstring(' '.join(test[0].text), dtype=np.uint8)
  print(trX.shape)
  print(vaX.shape)
  print(testX.shape)

  data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
  def __init__(self, data, seq_len):
    super().__init__()
    self.data = data
    self.seq_len = seq_len

  def __getitem__(self, index):
    rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
    full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
    return full_seq.cuda()

  def __len__(self):
    return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))


# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

def train():
  for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
      loss = model(next(train_loader))
      loss.backward()

    # print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
      model.eval()
      with torch.no_grad():
        loss = model(next(val_loader))
        logging.info(f'validation loss: {loss.item()}')
      # save model
      model.save(f'./{args.results_dir}/{args.model}_{args.dataset}_ep_{i}.pt')
      logging.info(f'saved model at epoch {i}')

    if i % GENERATE_EVERY == 0:
      model.eval()
      inp = random.choice(val_dataset)[:-1]
      prime = decode_tokens(inp)
      logging.info(f'%s \n\n %s', (prime, '*' * 100))

      sample = model.generate(inp.to(DEVICE), GENERATE_LENGTH)
      output_str = decode_tokens(sample)
      logging.info(output_str)


# train()
