"""
Suggested XLA flags:

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2"

Suggested LIBTPU: After Nov 12

export TPU_LIBRARY_PATH=/workspaces/torch/_libtpu.so

"""
from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

import os
import torch_xla
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla import runtime as xr
from itertools import chain
from tqdm import tqdm


def main(num_layers: int, profile_name: str, spmd: bool, offload: bool,
         profile: bool):
  if spmd:
    # Sharding
    num_devices = xr.global_runtime_device_count()
    tensor_axis = 4
    fsdp_axis = num_devices // tensor_axis
    mesh_shape = (fsdp_axis, tensor_axis)
    print(f"Single-slice sharding: mesh={mesh_shape}")
    spmd_mesh = xs.Mesh(
        list(range(num_devices)), mesh_shape, ('fsdp', 'tensor'))
    xs.set_global_mesh(spmd_mesh)
    xr.use_spmd()

  print("Building model")
  device = torch_xla.device()
  config = DecoderOnlyConfig(hidden_size=1024, num_hidden_layers=num_layers)
  config.intermediate_size = 4096
  config.vocab_size = 8192
  model = DecoderOnlyModel(config=config).to(device)
  batch_size = 16
  sequence_length = 512

  model.use_offload_(offload)
  model.use_scan_(True)

  if spmd:
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
  optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
  torch_xla.sync(wait=True)

  def step_fn():
    optimizer.zero_grad()
    output = model(input_ids.clone())
    output.sum().backward()
    optimizer.step()

  compiled_step_fn = torch_xla.compile(
      step_fn, full_graph=True, name="train_step_fn")

  print("Compiling model")
  for i in range(2):
    compiled_step_fn()  # type:ignore
  torch_xla.sync(wait=True)

  model.zero_grad()
  torch_xla.sync(wait=True)

  # Start profiling
  if profile:
    print("Profiling model")
    logdir = f"profile/{profile_name}/"
    print(f"Log directory: {logdir}")
    os.makedirs(logdir, exist_ok=True)
    import torch_xla.debug.profiler as xp
    server = xp.start_server(9017)
    xp.trace_detached(
        service_addr="localhost:9017", logdir=logdir, duration_ms=10000)
  else:
    print("Running model")
  for i in tqdm(range(4)):
    compiled_step_fn()  # type:ignore
  torch_xla.sync(wait=True)
  if profile:
    del server

  print("Done!")

  print("XLA flags used:")
  print(os.getenv("LIBTPU_INIT_ARGS"))


if __name__ == "__main__":
  # Parse command-line arguments
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num-layers', type=int, default=30, help='Number of decoder layers')
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
  args = parser.parse_args()
  name = args.name
  main(
      args.num_layers,
      name,
      spmd=args.spmd,
      offload=args.offload,
      profile=args.profile)
