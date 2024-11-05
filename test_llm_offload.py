from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel
from offload_lib import offload

import torch_xla
import torch

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

import time
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)

# Offload the entire model, except the first embedding and the final norm.
# When the embedding layer is part of the offload wrapper, XLA complains that
# _xla_buffer_placement can only be specified on annotate_device_placement calls,
# despite the fact that we only use that attribute on annotate_device_placement calls.
# model = offload(model)
model.layers_sequential = offload(model.layers_sequential)

print("Compiling model")
for i in range(10):
  model.zero_grad()
  output = model(input_ids.clone())
  output.sum().backward()
  torch_xla.sync()
torch_xla.sync(wait=True)
model.zero_grad()
torch_xla.sync(wait=True)

# Start profiling
print("Profiling model")
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
