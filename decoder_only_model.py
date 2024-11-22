"""The decoder model taken from https://github.com/pytorch/xla/blob/master/examples/decoder_only_model.py

Adapted to support scan.
"""

from functools import partial
from typing import Tuple
from optree import tree_flatten, tree_map
import torch_xla.debug.profiler as xp
from torch_xla.experimental.scan_layers import scan_layers

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import torch.fx as fx
from torch import nn
from functorch.compile import min_cut_rematerialization_partition, default_partition, make_boxed_func  # type:ignore

import torch
import torch.autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import symbolic_trace
from functorch.compile import aot_function
from torch_xla.experimental.stablehlo_custom_call import place_to_host, place_to_device


# the default config is intentionally kept low to make it runable on a sigle tpu v2-8 core.
@dataclass
class DecoderOnlyConfig:
  hidden_size: int = 256
  num_hidden_layers: int = 5
  num_attention_heads: int = 8
  num_key_value_heads: int = 4
  intermediate_size: int = 32 * 256
  vocab_size: int = 3200
  use_flash_attention: bool = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :,
                                None, :, :].expand(batch, num_key_value_heads,
                                                   n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                               head_dim)


class RMSNorm(nn.Module):

  def __init__(self, hidden_size, eps=1e-6):
    """
    RMSNorm is equivalent to LlamaRMSNorm
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  @xp.trace_me("RMSNorm")
  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance +
                                                self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


# 1. no kv_cache
# 2. no rotary embedding
# 3. no attention_mask
class GroupQueryAttention(nn.Module):
  """Stripped-down version of the LlamaAttention"""

  def __init__(self, config: DecoderOnlyConfig):
    super().__init__()
    self.config = config

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    self.q_proj = nn.Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(
        self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(
        self.num_heads * self.head_dim, self.hidden_size, bias=False)
    self.flash_attention_impl = None

  @xp.trace_me("GroupQueryAttention")
  def forward(
      self,
      hidden_states: torch.Tensor,
  ) -> torch.Tensor:

    bsz, q_len, _ = hidden_states.size()
    # [B, S, H] -> [B, S, n_head * head_dim]
    query_states = self.q_proj(hidden_states)
    # [B, S, H] -> [B, S, n_kv_head * head_dim]
    key_states = self.k_proj(hidden_states)
    # [B, S, H] -> [B, S, n_kv_head * head_dim]
    value_states = self.v_proj(hidden_states)

    # [B, S, n_head * head_dim] -> [B, n_head, S, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    # [B, S, n_kv_head * head_dim] -> [B, n_kv_head, S, head_dim]
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    # [B, S, n_kv_head * head_dim] -> [B, n_kv_head, S, head_dim]
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    # [B, n_kv_head, S, head_dim] -> [B, n_head, S, head_dim]
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    # [B, n_kv_head, S, head_dim] -> [B, n_head, S, head_dim]
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if not self.config.use_flash_attention:
      # [B, n_head, S, head_dim] @ T([B, n_head, S, head_dim]) -> [B, n_head, S, S]
      attn_weights = torch.einsum('bnsh,bnkh->bnsk', query_states,
                                  key_states) / math.sqrt(self.head_dim)

      # upcast attention to fp32
      attn_weights = nn.functional.softmax(
          attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

      # [B, n_head, S, S] @ T([B, n_head, S, head_dim]) -> [B, n_head, S, head_dim]
      attn_output = torch.einsum('bnsk,bnkh->bnsh', attn_weights, value_states)
    else:
      assert self.flash_attention_impl != None
      # [B, n_head, S, head_dim], [B, n_head, S, head_dim], [B, n_head, S, head_dim]
      # -> [B, n_head, S, head_dim]
      attn_output = self.flash_attention_impl(query_states, key_states,
                                              value_states)

    # [B, n_head, S, head_dim] -> [B * S * n_head * head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # [B * S * n_head * head_dim] -> [B, S, H]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # [B, S, H] -> [B, S, H]
    attn_output = self.o_proj(attn_output)

    return attn_output


class MLP(nn.Module):
  """Stripped-down version of the LlamaMLP"""

  def __init__(self, config: DecoderOnlyConfig):
    super().__init__()
    self.config = config

    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(
        self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(
        self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(
        self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = F.silu

  @xp.trace_me("MLP")
  def forward(self, x):
    # [B, S, H] -> [B, S, I]
    up_proj = self.up_proj(x)
    # [B, S, H] -> [B, S, I]
    gate_proj = self.act_fn(self.gate_proj(x))
    # ([B, S, I] * [B, S, I]) -> [B, S, H]
    down_proj = self.down_proj(gate_proj * up_proj)
    return down_proj


class DecoderLayer(nn.Module):

  def __init__(self, config: DecoderOnlyConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = GroupQueryAttention(config=config)
    self.mlp = MLP(config)
    self.input_layernorm = RMSNorm(config.hidden_size)
    self.post_attention_layernorm = RMSNorm(config.hidden_size)

  @xp.trace_me("DecoderLayer")
  def forward(
      self,
      hidden_states: torch.Tensor,
  ) -> torch.Tensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = self.self_attn(hidden_states=hidden_states,)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# 1. no gradient_checkpointing
# 2. no padding_idx
# 3. no kv cache
class DecoderOnlyModel(nn.Module):

  def __init__(self, config: DecoderOnlyConfig):
    super(DecoderOnlyModel, self).__init__()
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
        [DecoderLayer(config) for _ in range(config.num_hidden_layers)])
    self.norm = RMSNorm(config.hidden_size)
    self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
    self.use_scan = False
    self.use_offload = False

  def use_offload_(self, use_offload: bool):
    self.use_offload = use_offload

  def use_scan_(self, use_scan: bool):
    self.use_scan = use_scan

  @xp.trace_me("DecoderOnlyModel")
  def forward(
      self,
      input_ids: torch.Tensor,
  ) -> torch.Tensor:
    inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    if self.use_scan:
      hidden_states = scan_layers(
          self.layers,
          hidden_states,
          partition_fn=custom_partition_fn
          if self.use_offload else default_partition)
    else:
      for layer in self.layers:
        hidden_states = layer(hidden_states)

    hidden_states = self.norm(hidden_states)
    # [B, S, H] -> [B, S, V]
    return self.lm_head(hidden_states)


def custom_partition_fn(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
):
  fwd, bwd = min_cut_rematerialization_partition(
      joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
  with torch.device('meta'):
    fw_example_args = make_arguments(fwd)
    bw_example_args = make_arguments(bwd)

  # TODO: ensure we remat all and only save decoder inputs.
  # TODO: offload the decoder inputs once we replicate torch.utils checkpointing.
  with torch.no_grad():

    def forward(**kwargs):
      # TODO: cannot offload model weights. model weights will be permuted/all-gathered.
      # If model weights is on host, that's not supported. We need to identify which
      # tensors are model weights and skip offloading them.
      print("Forward is called by AOTAutograd tracing.")
      import pdb
      pdb.set_trace()
      out = fwd(**kwargs)
      return (out[0],) + tuple(
          torch.ops.xla.place_to_host(v) for v in out[1:])  # type:ignore

    def backward(**kwargs):
      kwargs = {
          k: torch.ops.xla.place_to_device(v)  # type: ignore
          for k, v in kwargs.items()
      }
      return bwd(**kwargs)

    # Use AOTAutograd to retrace forward
    graph = [None]

    def get_graph(g, _):
      graph[0] = g
      # print("Got graph: ")
      # print(g.code)
      return make_boxed_func(g)

    # print("AOT tracing the forward")
    _ = aot_function(forward, fw_compiler=get_graph)(**fw_example_args)
    aot_forward = graph[0]

    # print("AOT tracing the backward")
    _ = aot_function(backward, fw_compiler=get_graph)(**bw_example_args)
    aot_backward = graph[0]

    return aot_forward, aot_backward


def make_arguments(gm):
  example_args = {}
  for node in gm.graph.nodes:
    if node.op != 'placeholder':
      continue
    if 'tensor_meta' in node.meta:
      tensor_meta = node.meta['tensor_meta']
      # print(f"Node: {node.name}, Shape: {tensor_meta.shape}")
      tensor = torch.zeros(
          tensor_meta.shape,
          dtype=tensor_meta.dtype,
          requires_grad=tensor_meta.requires_grad)
      example_args[node.name] = tensor
  return example_args


########## Host offloading ops ###########


@torch.library.custom_op("xla::place_to_host", mutates_args=())
def to_host(t: torch.Tensor) -> torch.Tensor:
  if t is None:
    return None
  return place_to_host(t)


@to_host.register_fake
def _(t: torch.Tensor) -> torch.Tensor:
  if t is None:
    return None
  return torch.empty_like(t)


def to_host_backward(ctx, grad):
  return grad


to_host.register_autograd(to_host_backward)


@torch.library.custom_op("xla::place_to_device", mutates_args=())
def to_device(t: torch.Tensor) -> torch.Tensor:
  if t is None:
    return None
  return place_to_device(t)


@to_device.register_fake
def _(t: torch.Tensor) -> torch.Tensor:
  if t is None:
    return None
  return torch.empty_like(t)


def to_device_backward(ctx, grad):
  return grad


to_device.register_autograd(to_device_backward)
