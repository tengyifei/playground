"""
JAX implementation of a Decoder-Only Model with Training Profiling.

This script mirrors the functionality of the PyTorch script,
leveraging Flax for model definitions and Optax for optimization.
Profiling is handled using TensorBoard's profiler.

Includes a '--scan' command-line argument to use flax.linen.scan
for iterating over decoder layers when specified.
"""

import argparse  # For command-line argument parsing
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax.linen import remat
from typing import Any, Callable, Sequence
import optax
import numpy as np
from dataclasses import dataclass
from functools import partial
import math
import time
import os
from tensorboardX import SummaryWriter

# ----------------------------
# Configuration
# ----------------------------


@dataclass
class DecoderOnlyConfig:
  hidden_size: int = 256
  num_hidden_layers: int = 5
  num_attention_heads: int = 8
  num_key_value_heads: int = 4
  intermediate_size: int = 32 * 256
  vocab_size: int = 3200
  use_flash_attention: bool = False


# ----------------------------
# Model Components
# ----------------------------


class RMSNorm(nn.Module):
  hidden_size: int
  eps: float = 1e-6

  @nn.compact
  def __call__(self, hidden_states):
    variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
    weight = self.param('weight', nn.initializers.ones, (self.hidden_size,))
    return hidden_states * weight


class GroupQueryAttention(nn.Module):
  config: DecoderOnlyConfig

  def setup(self):
    config = self.config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False)
    self.k_proj = nn.Dense(
        self.num_key_value_heads * self.head_dim, use_bias=False)
    self.v_proj = nn.Dense(
        self.num_key_value_heads * self.head_dim, use_bias=False)
    self.o_proj = nn.Dense(self.hidden_size, use_bias=False)

  def repeat_kv(self, hidden_states, n_rep):
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
      return hidden_states
    hidden_states = jnp.expand_dims(
        hidden_states, axis=2)  # [B, num_kv_heads, 1, S, head_dim]
    hidden_states = jnp.tile(
        hidden_states,
        (1, 1, n_rep, 1, 1))  # [B, num_kv_heads, n_rep, S, head_dim]
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

  def __call__(self, hidden_states):
    bsz, q_len, _ = hidden_states.shape
    config = self.config

    # Project inputs to query, key, and value
    query_states = self.q_proj(hidden_states)  # [B, S, n_head * head_dim]
    key_states = self.k_proj(hidden_states)  # [B, S, n_kv_head * head_dim]
    value_states = self.v_proj(hidden_states)  # [B, S, n_kv_head * head_dim]

    # Reshape and transpose for multi-head attention
    query_states = query_states.reshape(bsz, q_len, config.num_attention_heads,
                                        self.head_dim)
    query_states = jnp.transpose(query_states,
                                 (0, 2, 1, 3))  # [B, n_head, S, head_dim]

    key_states = key_states.reshape(bsz, q_len, config.num_key_value_heads,
                                    self.head_dim)
    key_states = jnp.transpose(key_states,
                               (0, 2, 1, 3))  # [B, n_kv_head, S, head_dim]

    value_states = value_states.reshape(bsz, q_len, config.num_key_value_heads,
                                        self.head_dim)
    value_states = jnp.transpose(value_states,
                                 (0, 2, 1, 3))  # [B, n_kv_head, S, head_dim]

    # Repeat key and value states
    key_states = self.repeat_kv(
        key_states, self.num_key_value_groups)  # [B, n_head, S, head_dim]
    value_states = self.repeat_kv(
        value_states, self.num_key_value_groups)  # [B, n_head, S, head_dim]

    # Scaled Dot-Product Attention
    scaling = 1.0 / math.sqrt(self.head_dim)
    attn_weights = jnp.einsum('bnsh,bnkh->bnsk', query_states,
                              key_states) * scaling  # [B, n_head, S, S]
    attn_weights = nn.softmax(
        attn_weights, axis=-1).astype(query_states.dtype)  # [B, n_head, S, S]

    attn_output = jnp.einsum('bnsk,bnkh->bnsh', attn_weights,
                             value_states)  # [B, n_head, S, head_dim]

    # Reshape back to [B, S, H]
    attn_output = jnp.transpose(attn_output,
                                (0, 2, 1, 3)).reshape(bsz, q_len,
                                                      self.hidden_size)

    # Output projection
    attn_output = self.o_proj(attn_output)  # [B, S, H]
    return attn_output


class MLP(nn.Module):
  config: DecoderOnlyConfig

  def setup(self):
    config = self.config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size

    self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False)
    self.up_proj = nn.Dense(self.intermediate_size, use_bias=False)
    self.down_proj = nn.Dense(self.hidden_size, use_bias=False)

  def __call__(self, x):
    up_proj = self.up_proj(x)  # [B, S, I]
    gate_proj = nn.silu(self.gate_proj(x))  # [B, S, I]
    down_proj = self.down_proj(gate_proj * up_proj)  # [B, S, H]
    return down_proj


# Import offloadable from partial_eval for host offloading
from jax._src.interpreters import partial_eval as pe
global_offload = False


def offload_all(prim, *_, **params):
  if global_offload:
    return pe.Offloadable(src="device", dst="pinned_host")
  else:
    return False  # Do not offload during initialization


class DecoderLayer(nn.Module):
  config: DecoderOnlyConfig

  def setup(self):
    self.self_attn = GroupQueryAttention(config=self.config)
    self.mlp = MLP(config=self.config)
    self.input_layernorm = RMSNorm(hidden_size=self.config.hidden_size)
    self.post_attention_layernorm = RMSNorm(hidden_size=self.config.hidden_size)

  # The following specifies remat with host offloading.
  @partial(nn.remat, policy=offload_all)  # type: ignore
  def __call__(self, hidden_states, _):
    # '_' is ignored (needed for scanning)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, None  # Return None for the scan carry


class DecoderOnlyModel(nn.Module):
  config: DecoderOnlyConfig
  use_scan: bool = False  # Flag to control scanning

  def setup(self):
    self.embed_tokens = nn.Embed(
        num_embeddings=self.config.vocab_size, features=self.config.hidden_size)

    if self.use_scan:
      # Wrap DecoderLayer with nn.scan
      self.layers = nn.scan(
          DecoderLayer,
          variable_axes={'params': 0},
          split_rngs={'params': True},
          length=self.config.num_hidden_layers,
      )(
          config=self.config)
    else:
      self.layers = [
          DecoderLayer(config=self.config)
          for _ in range(self.config.num_hidden_layers)
      ]
    self.norm = RMSNorm(hidden_size=self.config.hidden_size)
    self.output = nn.Dense(self.config.vocab_size, use_bias=False)

  def __call__(self, input_ids):
    hidden_states = self.embed_tokens(input_ids)  # [B, S, H]

    if self.use_scan:
      print("Use scan")
      assert isinstance(self.layers, DecoderLayer)
      xs = [None] * self.config.num_hidden_layers  # xs is a list of None
      hidden_states, _ = self.layers(hidden_states, xs)
    else:
      # Pass through decoder layers
      print("Use for loop")
      assert isinstance(self.layers, list)
      for layer in self.layers:
        hidden_states, _ = layer(hidden_states, None)

    hidden_states = self.norm(hidden_states)
    logits = self.output(hidden_states)  # [B, S, V]
    return logits


# ----------------------------
# Training Utilities
# ----------------------------


def create_train_state(rng, model, config):
  params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))['params']
  tx = optax.adam(learning_rate=1e-3)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


def compute_loss(params, apply_fn, input_ids):
  # Call apply_fn directly with the correct arguments
  logits = apply_fn({'params': params}, input_ids)  # [B, S, V]

  # Shift targets for cross-entropy calculation
  targets = input_ids[:, 1:]
  logits = logits[:, :-1, :]  # Align shapes [B, S-1, V]

  # Compute cross-entropy loss
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
  return loss


@jax.jit
def train_step(state, input_ids):
  # Loss function with only the params being differentiable
  def loss_fn(params):
    return compute_loss(params, state.apply_fn, input_ids)

  # Compute the loss and its gradient with respect to parameters
  loss, grads = jax.value_and_grad(loss_fn)(state.params)

  # Apply the gradients to update the model's parameters
  state = state.apply_gradients(grads=grads)

  return state, loss


# ----------------------------
# Profiling Setup
# ----------------------------


def setup_tensorboard_profiler(logdir):
  writer = SummaryWriter(logdir=logdir)
  return writer


# ----------------------------
# Main Training Loop
# ----------------------------


def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--scan', action='store_true', help='Use scan in the decoder layers')
  args = parser.parse_args()

  config = DecoderOnlyConfig(
      hidden_size=1024,
      num_hidden_layers=20,
      intermediate_size=4096,
      vocab_size=8192)

  global global_offload
  global_offload = False  # Disable offloading during initialization

  # Initialize model
  rng = jax.random.PRNGKey(0)
  model = DecoderOnlyModel(config=config, use_scan=args.scan)
  state = create_train_state(rng, model, config)

  # Define training parameters
  batch_size = 16
  sequence_length = 512

  # Generate random input_ids
  input_rng = jax.random.PRNGKey(1)
  input_ids = jax.random.randint(
      input_rng, (batch_size, sequence_length),
      0,
      config.vocab_size,
      dtype=jnp.int32)

  # Initialize TensorBoard writer for profiling
  logdir = "profile/"
  writer = setup_tensorboard_profiler(logdir)

  # Compile the model by running a few training steps
  global_offload = True  # Enable offloading during training
  print("Compiling model...")
  for _ in range(10):
    state, loss = train_step(state, input_ids)
  print("Compilation done.")

  # Start profiling
  print("Starting profiling...")
  jax.profiler.start_trace("profile/")
  for step in range(10):
    state, loss = train_step(state, input_ids)
    writer.add_scalar('loss', float(loss), step)
    if step % 5 == 0:
      print(f"Step {step}, Loss: {loss}")
  jax.profiler.stop_trace()
  print("Profiling done.")

  writer.close()
  print("Done!")


if __name__ == "__main__":
  main()
