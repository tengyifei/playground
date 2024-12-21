"""
Suggested XLA flags:

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2"

Suggested LIBTPU: After Nov 12

How to use custom libtpu:

```
export TPU_LIBRARY_PATH=/workspaces/torch/_libtpu.so
```

Examples:

```
# Test 80 layer toy decoder with SPMD distribution
./test_llm_offload.sh --profile --num-layers 80 --spmd --scan

# Test 80 layer toy decoder with SPMD distribution and also offload decoder inputs
./test_llm_offload.sh --profile --num-layers 80 --spmd --scan --offload

# Test 80 layer toy decoder with SPMD distribution and also offload decoder inputs and with flash attention
./test_llm_offload.sh --profile --num-layers 80 --spmd ---scan --offload --flash-attention
```

"""
from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel
from aot_flash_attention import flash_attention_2

import math
import time
import os
import torch_xla
import torch_xla.debug.metrics
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
from torch_xla import runtime as xr
from itertools import chain
from tqdm import tqdm

from transformers.optimization import Adafactor


def main(num_layers: int, profile_name: str, num_steps: int, spmd: bool,
         offload: bool, profile: bool, flash_attention: bool, scan: bool):
  if spmd:
    # Sharding
    num_devices = xr.global_runtime_device_count()
    tensor_axis = 2
    fsdp_axis = num_devices // tensor_axis
    mesh_shape = (fsdp_axis, tensor_axis)
    print(f"Single-slice sharding: mesh={mesh_shape}")
    spmd_mesh = xs.Mesh(
        list(range(num_devices)), mesh_shape, ('fsdp', 'tensor'))
    xs.set_global_mesh(spmd_mesh)
    xr.use_spmd()
  else:
    spmd_mesh = None

  print("Building model")
  device = torch_xla.device()
  config = DecoderOnlyConfig(
      hidden_size=2048,
      num_hidden_layers=num_layers,
      use_flash_attention=flash_attention)
  config.intermediate_size = 8192
  config.vocab_size = 16384
  model = DecoderOnlyModel(config=config).bfloat16().to(device)
  batch_size = 32
  sequence_length = 4096

  model.use_offload_(offload)
  model.use_scan_(scan)
  if flash_attention:
    assert spmd, "Flash attention requires SPMD"
    for layer in model.layers:
      layer.self_attn.flash_attention_impl = flash_attention_2  # type: ignore
      # from torch_xla.experimental.custom_kernel import flash_attention
    # layer.self_attn.flash_attention_impl = flash_attention  # type: ignore

  if spmd and spmd_mesh:
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

  # Generate random input_ids within the range of the vocabulary size
  input_ids = torch.randint(
      0, config.vocab_size, (batch_size, sequence_length), device=device)
  # Shard the input data too.
  if spmd and spmd_mesh:
    xs.mark_sharding(input_ids, spmd_mesh, ('fsdp', None))
    xs.set_global_mesh(spmd_mesh)

  optimizer = Adafactor(
      model.parameters(),
      lr=0.00001,
      scale_parameter=False,
      relative_step=False)
  torch_xla.sync(wait=True)

  def loss_fn(logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)

  def step_closure(loss):
    loss_str = str(loss.item() if loss is not None else "none")
    bar.set_postfix_str("loss=" + loss_str)
    if math.isnan(loss.item()):
      raise RuntimeError("Loss became NaN")

  def step_fn(bar=None):
    optimizer.zero_grad()
    logits = model(input_ids).float()
    loss = loss_fn(logits, input_ids)
    with xp.Trace('backward'):
      loss.backward()
    with xp.Trace('optimizer'):
      optimizer.step()
    if bar:
      torch_xla.xm.add_step_closure(step_closure, args=(loss,))

  compiled_step_fn = torch_xla.compile(
      step_fn, full_graph=True, name="train_step")

  print("Compiling model")
  start = time.time()
  for _ in range(2):
    compiled_step_fn()  # type:ignore
  torch_xla.sync(wait=True)
  end = time.time()
  print(f"Compilation took: {end - start} seconds")

  model.zero_grad()
  torch_xla.sync(wait=True)

  # Set a debug env var.
  os.environ["DEBUG_TRANSFER_IR_VALUE_TENSOR_TO_XLA_DATA"] = "1"

  # Start profiling
  server = None
  if profile:
    print("Profiling model")
    logdir = f"profile/{profile_name}/"
    print(f"Log directory: {logdir}")
    os.makedirs(logdir, exist_ok=True)
    server = xp.start_server(9017)
    xp.trace_detached(
        service_addr="localhost:9017", logdir=logdir, duration_ms=10000)
  else:
    print("Running model")
  bar = tqdm(range(num_steps))
  for i in bar:
    compiled_step_fn(bar)  # type:ignore
  torch_xla.sync(wait=True)
  if profile:
    del server

  print("Done!")

  print("XLA metrics:")
  print(torch_xla.debug.metrics.metrics_report())

  print("XLA flags used:")
  print(os.getenv("LIBTPU_INIT_ARGS"))


if __name__ == "__main__":
  # Parse command-line arguments
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num-layers', type=int, default=30, help='Number of decoder layers')
  parser.add_argument(
      '--num-steps', type=int, default=10, help='Number of train steps')
  parser.add_argument('--name', type=str, required=True, help='Name of the run')
  parser.add_argument(
      '--spmd', action='store_true', required=False, help='Use SPMD')
  parser.add_argument(
      '--offload',
      action='store_true',
      required=False,
      help='Use host offloading')
  parser.add_argument(
      '--profile', action='store_true', required=False, help='Profile model')
  parser.add_argument(
      '--flash-attention',
      action='store_true',
      required=False,
      help='Use flash attention')
  parser.add_argument(
      '--scan', action='store_true', required=False, help='Use scan')
  args = parser.parse_args()
  name = args.name
  main(
      args.num_layers,
      name,
      num_steps=args.num_steps,
      spmd=args.spmd,
      offload=args.offload,
      profile=args.profile,
      flash_attention=args.flash_attention,
      scan=args.scan)
