import copy
from math import ceil
from functools import partial
import math
from random import randrange
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn, einsum

from einops import rearrange, repeat

from gpt_models.reversible import ReversibleSequence, SequentialSequence

# functions


def exists(val):
  return val is not None


def cast_tuple(val, num):
  return ((val,) * num) if not isinstance(val, tuple) else val


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
  seqlen = tensor.shape[dim]
  m = seqlen / multiple
  if m.is_integer():
    return tensor
  remainder = ceil(m) * multiple - seqlen
  pad_offset = (0,) * (-1 - dim) * 2
  return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def dropout_layers(layers, prob_survival):
  if prob_survival == 1:
    return layers

  num_layers = len(layers)
  to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

  # make sure at least one layer makes it
  if all(to_drop):
    rand_index = randrange(num_layers)
    to_drop[rand_index] = False

  layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
  return layers

# helper classes


class Conv1D(nn.Module):
  def __init__(self, nf, nx):
    super(Conv1D, self).__init__()
    self.nf = nf
    w = torch.empty(nx, nf)
    nn.init.normal_(w, std=0.02)
    self.weight = Parameter(w)
    self.bias = Parameter(torch.zeros(nf))

  def forward(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    x = x.view(*size_out)
    return x


class MLP(nn.Module):
  def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
    super(MLP, self).__init__()
    nx = config.n_embd
    self.c_fc = Conv1D(n_state, nx)
    self.c_proj = Conv1D(nx, n_state)
    self.act = F.gelu

  def forward(self, x):
    h = self.act(self.c_fc(x))
    h2 = self.c_proj(h)
    return h2


class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x):
    return self.fn(x) + x


class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)

  def forward(self, x, **kwargs):
    x = self.norm(x)
    return self.fn(x, **kwargs)


class GEGLU(nn.Module):
  def forward(self, x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class FeedForward(nn.Module):
  def __init__(self, dim, mult=4):
    super().__init__()
    inner_dim = int(dim * mult * 2 / 3)

    self.net = nn.Sequential(
        nn.Linear(dim, inner_dim * 2),
        GEGLU(),
        nn.Linear(inner_dim, dim)
    )

  def forward(self, x):
    return self.net(x)


class Attention(nn.Module):
  def __init__(self, dim_in, dim_out, dim_inner):
    super().__init__()
    self.scale = dim_inner ** -0.5
    self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
    self.to_out = nn.Linear(dim_inner, dim_out)

  def forward(self, x):
    device = x.device
    q, k, v = self.to_qkv(x).chunk(3, dim=-1)
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
    sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return self.to_out(out)


class LocalAttention(nn.Module):
  def __init__(self, dim_in, dim_inner, dim_out, window=128):
    super().__init__()
    self.scale = dim_inner ** -0.5
    self.window = window

    self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
    self.to_out = nn.Linear(dim_inner, dim_out)

  def forward(self, x):
    b, n, *_, device, w = *x.shape, x.device, self.window

    x = pad_to_multiple(x, w, dim=-2, value=0.)
    q, k, v = self.to_qkv(x).chunk(3, dim=-1)

    def window_fn(t): return rearrange(t, 'b (w n) d -> b w n d', n=w)
    q, k, v = map(window_fn, (q, k, v))

    k, v = map(lambda t: F.pad(t, (0, 0, 0, 0, 1, 0)), (k, v))
    k, v = map(lambda t: torch.cat((k[:, :-1], k[:, 1:]), dim=2), (k, v))

    sim = einsum('b w i d, b w j d -> b w i j', q, k) * self.scale
    buckets, i, j = sim.shape[-3:]

    mask_value = -torch.finfo(sim.dtype).max
    mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
    mask = repeat(mask, 'i j -> () u i j', u=buckets)

    sim.masked_fill_(mask, mask_value)

    attn = sim.softmax(dim=-1)

    out = einsum('b w i j, b w j d -> b w i d', attn, v)
    out = rearrange(out, 'b w n d -> b (w n) d')
    out = self.to_out(out[:, :n])
    return out


class SelfAttention(nn.Module):
  def __init__(self, nx, n_ctx, config, scale=False):
    super(SelfAttention, self).__init__()
    n_state = nx  # in Attention: n_state=768 (nx=n_embd)
    # [switch nx => n_state from Block to Attention to keep identical to TF implem]
    assert n_state % config.n_head == 0
    self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
    self.n_head = config.n_head
    self.split_size = n_state
    self.scale = scale
    self.c_attn = Conv1D(n_state * 3, nx)
    self.c_proj = Conv1D(n_state, nx)

  def _attn(self, q, k, v):
    w = torch.matmul(q, k)
    if self.scale:
      w = w / math.sqrt(v.size(-1))
    nd, ns = w.size(-2), w.size(-1)
    b = self.bias[:, :, ns-nd:ns, :ns]
    w = w * b - 1e10 * (1 - b)
    w = nn.Softmax(dim=-1)(w)
    return torch.matmul(w, v)

  def merge_heads(self, x):
    x = x.permute(0, 2, 1, 3).contiguous()
    new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
    return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

  def split_heads(self, x, k=False):
    new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
    x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
    if k:
      return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
    else:
      return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

  def forward(self, x, layer_past=None):
    x = self.c_attn(x)
    query, key, value = x.split(self.split_size, dim=2)
    query = self.split_heads(query)
    key = self.split_heads(key, k=True)
    value = self.split_heads(value)
    if layer_past is not None:
      # transpose back cf below
      past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
      key = torch.cat((past_key, key), dim=-1)
      value = torch.cat((past_value, value), dim=-2)
    # transpose to have same shapes for stacking
    present = torch.stack((key.transpose(-2, -1), value))
    a = self._attn(query, key, value)
    a = self.merge_heads(a)
    a = self.c_proj(a)
    return a, present


class CausalSGU(nn.Module):
  def __init__(
      self,
      dim,
      dim_seq,
      init_eps=1e-3,
      heads=4,
      act=nn.Identity()
  ):
    super().__init__()
    dim_out = dim // 2

    self.norm = nn.LayerNorm(dim_out)

    self.heads = heads
    self.weight = nn.Parameter(torch.zeros(heads, dim_seq, dim_seq))
    self.bias = nn.Parameter(torch.zeros(heads, dim_seq))

    init_eps /= dim_seq
    nn.init.uniform_(self.weight, -init_eps, init_eps)
    nn.init.constant_(self.bias, 1.)

    self.act = act
    self.register_buffer('mask', ~torch.ones(dim_seq, dim_seq).triu_(1).bool())

  def forward(self, x, gate_res=None):
    device, n, h = x.device, x.shape[1], self.heads

    res, gate = x.chunk(2, dim=-1)
    gate = self.norm(gate)

    weight, bias = self.weight, self.bias
    weight, bias = weight[:, :n, :n], bias[:, :n]

    weight = weight * self.mask[None, :n, :n].int().float()

    gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)
    gate = einsum('b h n d, h m n -> b h m d', gate, weight)
    gate = gate + rearrange(bias, 'h n -> () h n ()')
    gate = rearrange(gate, 'b h n d -> b n (h d)')

    if exists(gate_res):
      gate = gate + gate_res

    return self.act(gate) * res


class CausalLocalSGU(nn.Module):
  def __init__(
      self,
      dim,
      dim_seq,
      init_eps=1e-3,
      heads=4,
      window=128,
      act=nn.Identity()
  ):
    super().__init__()
    dim_out = dim // 2

    self.norm = nn.LayerNorm(dim_out)

    self.heads = heads
    self.window = window
    self.weight = nn.Parameter(torch.zeros(heads, window, window * 2))
    self.bias = nn.Parameter(torch.zeros(heads, window))

    init_eps /= window
    nn.init.uniform_(self.weight, -init_eps, init_eps)
    nn.init.constant_(self.bias, 1.)

    self.act = act
    self.register_buffer('mask', ~torch.ones(window, window * 2).triu_(window + 1).bool())

  def forward(self, x, gate_res=None):
    device, n, h, w = x.device, x.shape[1], self.heads, self.window

    res, gate = x.chunk(2, dim=-1)

    gate = pad_to_multiple(gate, w, dim=-2)
    gate = rearrange(gate, 'b (w n) d -> b w n d', n=w)

    gate = self.norm(gate)

    gate = F.pad(gate, (0, 0, 0, 0, 1, 0), value=0.)
    gate = torch.cat((gate[:, :-1], gate[:, 1:]), dim=2)

    weight, bias = self.weight, self.bias

    weight = weight * self.mask[None, ...].int().float()

    gate = rearrange(gate, 'b w n (h d) -> b w h n d', h=h)
    gate = einsum('b w h n d, h m n -> b w h m d', gate, weight)
    gate = gate + rearrange(bias, 'h n -> () () h n ()')

    gate = rearrange(gate, 'b w h n d -> b w n (h d)')

    gate = rearrange(gate, 'b w n d -> b (w n) d')
    gate = gate[:, :n]

    if exists(gate_res):
      gate = gate + gate_res

    return self.act(gate) * res


class AxiallyFold(nn.Module):
  def __init__(self, dim, every, fn):
    super().__init__()
    self.fn = fn
    self.every = every
    self.conv = nn.Conv1d(dim, dim, kernel_size=every, groups=dim) if every > 1 else None

  def forward(self, x):
    every = self.every
    if every <= 1:
      return self.fn(x)

    n = x.shape[1]
    x = pad_to_multiple(x, self.every, dim=-2)
    x = rearrange(x, 'b (n e) d -> (b e) n d', e=every)
    x = self.fn(x)

    x = rearrange(x, '(b e) n d -> b d (n e)', e=every)
    x = F.pad(x, (every - 1, 0), value=0)
    out = self.conv(x)
    out = rearrange(out, 'b d n -> b n d')
    return out[:, :n]


class gMLPBlock(nn.Module):
  def __init__(
      self,
      *,
      dim,
      seq_len,
      dim_ff,
      heads=4,
      causal=False,
      window=None,
      attn_dim=None,
      act=nn.Identity()
  ):
    super().__init__()
    is_windowed = exists(window) and window < seq_len

    SGU_klass = partial(CausalLocalSGU, window=window) if is_windowed else CausalSGU
    Attention_klass = partial(LocalAttention, window=window) if is_windowed else Attention

    self.attn = Attention_klass(dim_in=dim, dim_inner=attn_dim,
                                dim_out=dim_ff // 2) if exists(attn_dim) else None

    self.proj_in = nn.Sequential(
        nn.Linear(dim, dim_ff),
        nn.GELU()
    )
    self.sgu = SGU_klass(dim_ff, seq_len, causal, heads=heads, act=act)
    self.proj_out = nn.Linear(dim_ff // 2, dim)

  def forward(self, x):
    gate_res = self.attn(x) if exists(self.attn) else None
    x = self.proj_in(x)
    x = self.sgu(x, gate_res=gate_res)
    x = self.proj_out(x)
    return x


class Block(nn.Module):
  def __init__(self, n_ctx, config, scale=False):
    super(Block, self).__init__()
    nx = config.n_embd
    self.ln_1 = nn.LayerNorm(nx, eps=1e-5)
    self.attn = SelfAttention(nx, n_ctx, config, scale)
    self.ln_2 = nn.LayerNorm(nx, eps=1e-5)
    self.mlp = MLP(4 * nx, config)

  def forward(self, x, layer_past=None):
    a, present = self.attn(self.ln_1(x), layer_past=layer_past)
    x = x + a
    m = self.mlp(self.ln_2(x))
    x = x + m
    return x, present


# main classes

class SAGPT(nn.Module):
  def __init__(
      self,
      *,
      num_tokens,
      dim,
      seq_len,
      prob_survival=1.,
  ):
    super().__init__()
    self.seq_len = seq_len
    self.prob_survival = prob_survival

    config = GPT2Config(n_embd=dim)
    self.to_embed = nn.Embedding(num_tokens, dim)

    self.wte = nn.Embedding(num_tokens, dim)
    self.wpe = nn.Embedding(seq_len, dim)
    block = Block(config.n_ctx, config, scale=True)
    self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(num_tokens * 2, eps=1e-5)

  def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
    if past is None:
      past_length = 0
      past = [None] * len(self.h)
    else:
      past_length = past[0][0].size(-2)
    if position_ids is None:
      position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                  device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_ids.size(-1))
    position_ids = position_ids.view(-1, position_ids.size(-1))

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    if token_type_ids is not None:
      token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
      token_type_embeds = self.wte(token_type_ids)
    else:
      token_type_embeds = 0
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    presents = []
    for block, layer_past in zip(self.h, past):
      hidden_states, present = block(hidden_states, layer_past)
      presents.append(present)
    hidden_states = self.ln_f(hidden_states)
    output_shape = input_shape + (hidden_states.size(-1),)
    return hidden_states.view(*output_shape)


'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''


class GPT2Config(object):
  def __init__(
          self,
          vocab_size_or_config_json_file=50257,
          n_positions=1024,
          n_ctx=1024,
          n_embd=768,
          n_layer=16,
          n_head=16,
          layer_norm_epsilon=1e-5,
          initializer_range=0.02,
  ):
    self.vocab_size = vocab_size_or_config_json_file
    self.n_ctx = n_ctx
    self.n_positions = n_positions
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_range = initializer_range
