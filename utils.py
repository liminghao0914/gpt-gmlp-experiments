import torch
import os
import logging
from argparse import ArgumentParser


def parse_args():
  parser = ArgumentParser(description='gmlp-gpt')
  parser.add_argument('--dataset', '-d', type=str, default='enwik8')
  parser.add_argument('--model', '-m', type=str, default='gmlp')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--embed_dim', type=int, default=300)
  parser.add_argument('--d_hidden', type=int, default=200)
  parser.add_argument('--dp_ratio', type=int, default=0.2)
  parser.add_argument('--epochs', '-e', type=int, default=20)
  parser.add_argument('--lr', type=float, default=2e-4)
  parser.add_argument('--combine', type=str, default='cat')
  parser.add_argument('--results_dir', type=str, default='results')
  parser.add_argument('--test', '-t', type=str)
  return parser.parse_args()
