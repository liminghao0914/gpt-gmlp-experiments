import os
import sys

import torch

from torchtext.data import Field, Iterator
from torchtext import datasets

from pdb import set_trace

__all__ = ['multinli']



print("preparing the dataset for training...")
TEXT = Field(lower=True, tokenize='spacy',tokenizer_language = 'en_core_web_sm', batch_first=True)
LABEL = Field(sequential=False, unk_token=None, is_target=True)
print("dd")

train, dev, test = datasets.WikiText103.splits(TEXT)

print(len(train))
# TEXT.build_vocab(train, dev)
# LABEL.build_vocab(train)

for i in range(3):
    print('Example {}:'.format(i))
    print('Input text: {}'.format(' '.join(train[i].text)))
    print('Target text: {}'.format(' '.join(train[i+1].text)))


print("helo")