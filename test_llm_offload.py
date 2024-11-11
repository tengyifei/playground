from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

import torch_xla
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla import runtime as xr
from itertools import chain
from tqdm import tqdm

# Sharding
num_devices = xr.global_runtime_device_count()
tensor_axis = 4
fsdp_axis = num_devices // tensor_axis
mesh_shape = (fsdp_axis, tensor_axis)
print(f"Single-slice sharding: mesh={mesh_shape}")
spmd_mesh = xs.Mesh(list(range(num_devices)), mesh_shape, ('fsdp', 'tensor'))
xs.set_global_mesh(spmd_mesh)
xr.use_spmd()

print("Building model")
device = torch_xla.device()
config = DecoderOnlyConfig(hidden_size=1024, num_hidden_layers=120)
config.intermediate_size = 4096
config.vocab_size = 8192
model = DecoderOnlyModel(config=config).to(device)
batch_size = 16
sequence_length = 512

model.use_scan_(True)

# Mark model weights to be sharded
for name, param in chain(model.named_parameters(), model.named_buffers()):
  print('> [2D] Sharding tensor', name, param.shape)

  # Here we intentionally skip layernorm and moe.gate weights given they are small.
  if 'embed_tokens' in name:
    xs.mark_sharding(param, spmd_mesh, ('fsdp', 'tensor'))
  elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
    xs.mark_sharding(param, spmd_mesh, ('tensor', 'fsdp'))
  elif 'o_proj' in name:
    xs.mark_sharding(param, spmd_mesh, ('fsdp', 'tensor'))
  elif 'gate_proj' in name or 'up_proj' in name:
    xs.mark_sharding(param, spmd_mesh, ('tensor', 'fsdp'))
  elif 'down_proj' in name:
    xs.mark_sharding(param, spmd_mesh, ('fsdp', 'tensor'))
  elif 'lm_head' in name:
    xs.mark_sharding(param, spmd_mesh, (('tensor', 'fsdp'), None))

  print(f'{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

# Checkpointing
torch.xla = torch_xla.device()  # type:ignore

from types import MethodType
import torch.utils.checkpoint


def checkpoint_module(module):
  """
  Wrap a `module`'s `forward` method with gradient checkpointing (also called
  activation checkpointing) via `torch.utils.checkpoint.checkpoint`.
  """

  def _xla_checkpointed_forward_no_kwargs(m, num_args, num_kwargs,
                                          *packed_args):
    # unpack packed_args into args and kwargs
    assert num_args + num_kwargs * 2 == len(packed_args)
    args = packed_args[:num_args]
    kwargs = packed_args[num_args:]
    kwargs = dict(zip(kwargs[:num_kwargs], kwargs[num_kwargs:]))
    return m._xla_checkpointed_forward_original(*args, **kwargs)

  def _forward_with_checkpoint(m, *args, **kwargs):
    # pack args and kwargs together as `torch_xla.utils.checkpoint.checkpoint`
    # doesn't support keyword arguments
    packed_args = args + tuple(kwargs.keys()) + tuple(kwargs.values())
    input_requires_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in packed_args)
    if input_requires_grad:
      outputs = torch.utils.checkpoint.checkpoint(
          m._xla_checkpointed_forward_no_kwargs,
          len(args),
          len(kwargs),
          *packed_args,
          use_reentrant=False)
    else:
      # No input requires gradients so we won't checkpoint this forward pass.
      # Note that `m`` might have parameters that require gradients, but they
      # are beyond what `torch_xla.utils.checkpoint.checkpoint` can handle.
      outputs = m._xla_checkpointed_forward_original(*args, **kwargs)
    return outputs

  assert isinstance(module, torch.nn.Module)
  # replace `module`'s forward method with its checkpointed version
  module._xla_checkpointed_forward_original = module.forward  # type:ignore
  module._xla_checkpointed_forward_no_kwargs = MethodType(  # type:ignore
      _xla_checkpointed_forward_no_kwargs, module)
  module.forward = MethodType(_forward_with_checkpoint, module)
  return module


for i, block in enumerate(model.layers):
  model.layers[i] = checkpoint_module(block)

# Generate random input_ids within the range of the vocabulary size
input_ids = torch.randint(
    0, config.vocab_size, (batch_size, sequence_length), device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
torch_xla.sync(wait=True)

# Offload the entire model, except the first embedding and the final norm.
# When the embedding layer is part of the offload wrapper, XLA complains that
# _xla_buffer_placement can only be specified on annotate_device_placement calls,
# despite the fact that we only use that attribute on annotate_device_placement calls.
# model = offload(model)
# model.layers_sequential = offload(model.layers_sequential)

# Offload each layer.
# for i, block in enumerate(model.layers):
#   model.layers[i] = offload(block)
# model.layers_sequential = torch.nn.Sequential(*model.layers)


def step_fn():
  optimizer.zero_grad()
  output = model(input_ids.clone())
  output.sum().backward()
  optimizer.step()


compiled_step_fn = torch_xla.compile(
    step_fn, full_graph=True, name="train_step_fn")

print("Compiling model")
for i in range(2):
  compiled_step_fn()
torch_xla.sync(wait=True)

model.zero_grad()
torch_xla.sync(wait=True)

# Start profiling
print("Profiling model")
import torch_xla.debug.profiler as xp
server = xp.start_server(9017)
xp.trace_detached(
    service_addr="localhost:9017", logdir="profile/", duration_ms=60000)
for i in tqdm(range(3)):
  compiled_step_fn()
torch_xla.sync(wait=True)

print("Done!")

print("XLA flags used:")
import os
print(os.getenv("LIBTPU_INIT_ARGS"))
