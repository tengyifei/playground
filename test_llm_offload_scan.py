from torch_xla.experimental.stablehlo_custom_call import place_to_device, place_to_host
from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel
from torch.autograd.graph import saved_tensors_hooks

import torch_xla
import torch
import torch.nn as nn

print("Building model")
device = torch_xla.device()
config = DecoderOnlyConfig(hidden_size=1024, num_hidden_layers=30)
config.intermediate_size = 4096
config.vocab_size = 8192
model = DecoderOnlyModel(config=config).to(device)
batch_size = 16
sequence_length = 512

import gc
gc.collect()

# Generate random input_ids within the range of the vocabulary size
input_ids = torch.randint(
    0, config.vocab_size, (batch_size, sequence_length), device=device)
torch_xla.sync(wait=True)


class OffloadedModule(nn.Module):

  def __init__(self, m):
    super(OffloadedModule, self).__init__()
    self.m = m

  def forward(self, *args):

    def should_offload(t: torch.Tensor):
      for p in self.m.parameters():
        if t is p:
          print(f"Skip offloading {type(t)} {t.shape}")
          return False
      return True

    def maybe_place_to_host(t):
      if should_offload(t):
        print(f"Offload {t.shape} tensor to host")
        return place_to_host(t)
      else:
        return t

    def maybe_place_to_device(t):
      if should_offload(t):
        print(f"Bring back {t.shape} tensor to device")
        return place_to_device(t)
      else:
        return t

    def pack_fn(tensor: torch.Tensor):
      return maybe_place_to_host(tensor)

    def unpack_fn(input) -> torch.Tensor:
      return maybe_place_to_device(input)

    with saved_tensors_hooks(pack_fn, unpack_fn):
      out = self.m(*args)

    return out


# Wrap each decoder in an offload, then use scan to run the layers.
model.layers = nn.ModuleList([OffloadedModule(layer) for layer in model.layers])
model.use_scan_(True)

print("Compiling model")
for i in range(3):
  model.zero_grad()
  output = model(input_ids.clone())
  output.sum().backward()
  torch_xla.sync()
torch_xla.sync(wait=True)
model.zero_grad()
torch_xla.sync(wait=True)

# Start profiling
print("Profiling model")
import time
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)
xp.trace_detached(
    service_addr="localhost:9012", logdir="profile/", duration_ms=60000)
time.sleep(1)
for i in range(10):
  model.zero_grad()
  output = model(input_ids.clone())
  output.sum().backward()
  torch_xla.sync()
torch_xla.sync(wait=True)
model.zero_grad()
torch_xla.sync(wait=True)

print("Done!")

import os
print(os.getenv("LIBTPU_INIT_ARGS"))
