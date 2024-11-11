import torch_xla
import torch
from torch_xla import runtime as xr

from torch.autograd.graph import saved_tensors_hooks
from torch_xla.experimental.stablehlo_custom_call import (place_to_host,
                                                          place_to_device)


class OffloadingModule(torch.nn.Module):

  def __init__(self, m):
    super().__init__()
    self.m = m

  def forward(self, *args, **kwargs):
    with saved_tensors_hooks(place_to_host, place_to_device):
      return self.m(*args, **kwargs)


import decoder_only_model
from trainer import TrainDecoderOnlyBase
import functools

import torch
import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla import runtime as xr
from itertools import chain


class TrainDecoderOnlyFSDPv2(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__(
        decoder_only_model.DecoderOnlyConfig(
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=8192,
            vocab_size=16384,
        ))
    # Define the mesh following common SPMD practice
    num_devices = xr.global_runtime_device_count()
    tensor_axis = 4
    fsdp_axis = num_devices // tensor_axis
    mesh_shape = (fsdp_axis, tensor_axis)
    print(f"Single-slice sharding: mesh={mesh_shape}")
    spmd_mesh = xs.Mesh(
        list(range(num_devices)), mesh_shape, ('fsdp', 'tensor'))
    xs.set_global_mesh(spmd_mesh)

    model: decoder_only_model.DecoderOnlyModel = self.model  # type:ignore
    self.model = model

    # Mark model weights to be sharded
    for name, param in chain(model.named_parameters(), model.named_buffers()):
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

    # Shard the input.
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices
    train_loader = xu.SampleGenerator(
        data=(torch.randint(
            0,
            self.config.vocab_size, (self.batch_size, self.seq_len),
            dtype=torch.int64,
            device='cpu'),
              torch.randint(
                  0,
                  self.config.vocab_size, (self.batch_size, self.seq_len),
                  dtype=torch.int64,
                  device='cpu')),
        sample_count=self.train_dataset_len // self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(spmd_mesh,
                                       ('fsdp', None)))  # type:ignore

    # Apply checkpoint to each DecoderLayer layer.
    from torch_xla.distributed.fsdp import checkpoint_module
    for i, block in enumerate(self.model.layers):
      self.model.layers[i] = checkpoint_module(block)

    # Apply offloading to each DecoderLayer layer.
    from torch_xla.distributed.fsdp import checkpoint_module
    for i, block in enumerate(self.model.layers):
      self.model.layers[i] = OffloadingModule(block)

    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.00001)
    torch_xla.sync(wait=True)


xr.use_spmd()
base = TrainDecoderOnlyFSDPv2()

print("Compiling model")
base.num_steps = 3
base.start_training()
torch_xla.sync(wait=True)

print("Profiling model")
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)
xp.trace_detached(
    service_addr="localhost:9012", logdir="profile/", duration_ms=15000)
base.num_steps = 5
base.start_training()
torch_xla.sync(wait=True)
del server
