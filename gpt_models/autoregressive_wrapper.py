import math
import torch
from torch import nn
import torch.nn.functional as F

# helper function


def eval_decorator(fn):
  def inner(model, *args, **kwargs):
    was_training = model.training
    model.eval()
    out = fn(model, *args, **kwargs)
    model.train(was_training)
    return out
  return inner

# top k filtering


def top_k(logits, thres=0.9):
  k = int((1 - thres) * logits.shape[-1])
  val, ind = torch.topk(logits, k)
  probs = torch.full_like(logits, float('-inf'))
  probs.scatter_(1, ind, val)
  return probs


class AutoregressiveWrapper(nn.Module):
  def __init__(self, net, device=None, ignore_index=-100, pad_value=0):
    super().__init__()
    self.pad_value = pad_value
    self.ignore_index = ignore_index
    if device is None:
      self.net = net
    else:
      self.net = net.to(device)
      self.device = device
    self.max_seq_len = net.seq_len

  @torch.no_grad()
  @eval_decorator
  def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
    device = start_tokens.device
    num_dims = len(start_tokens.shape)

    if num_dims == 1:
      start_tokens = start_tokens[None, :]

    b, t = start_tokens.shape

    out = start_tokens

    for _ in range(seq_len):
      x = out[:, -self.max_seq_len:]
      logits = self.net(x, **kwargs)[:, -1, :]

      filtered_logits = top_k(logits, thres=filter_thres)
      probs = F.softmax(filtered_logits / temperature, dim=-1)

      sample = torch.multinomial(probs, 1)

      out = torch.cat((out, sample), dim=-1)

      if eos_token is not None and (sample == eos_token).all():
        break

    out = out[:, t:]

    if num_dims == 1:
      out = out.squeeze(0)

    return out

  def forward(self, x, **kwargs):
    # x = x.to(self.device)
    xi, xo = x[:, :-1], x[:, 1:]
    out = self.net(xi, **kwargs)
    loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
    return loss

  def entropy_loss(self, test_data, device, **kwargs):
    # x = x.to(self.device)
    losses = []
    with torch.no_grad():
      for x in test_data:
        xi, xo = x[:, :-1], x[:, 1:]
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index, reduction='none')
        losses.append(loss)
    losses = torch.cat(losses)
    return losses.mean().item()
  
  def bcp_loss(self, test_data, device, **kwargs):
    total_bits = 0
    total_chars = 0

    with torch.no_grad():
      for seq in test_data:
        seq_len = seq.shape[1]
        seq_tensor = seq[:, :-1]
        label = seq[:, 1:]
        output = self.net(seq_tensor,**kwargs)
        output = output.transpose(1, 2)
        loss = F.cross_entropy(output, label, ignore_index=self.ignore_index)
        bits = loss * seq_len
        total_bits += bits.item()
        total_chars += seq_len

    bpc = total_bits / (total_chars * torch.log2(torch.tensor([2.0], device=device)).item())
    # print(f'Bits per character: {bpc}')
    return bpc

  def perplexity(self, test_data, device, **kwargs):
    probs_list = []
    with torch.no_grad():
      for seq in test_data:
        seq_len = seq.shape[1]
        for _ in range(seq_len):
          x = seq[:, -self.max_seq_len:]
          logits = self.net(x, **kwargs)[:, -1, :]

          filtered_logits = top_k(logits)
          probs = F.softmax(filtered_logits, dim=-1)
        probs_list.append(torch.log(probs).mean())
    perplexity = torch.exp(torch.stack(probs_list).mean()).item()
    print(f'Perplexity: {perplexity}')
    # print(f'Bits per character: {bpc}')
    return perplexity
  
  def save(self, path):
    torch.save(self.net, path)