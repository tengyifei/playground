set -ex

mkdir -p profile
rm ir_dumps/scan-offload-ptxla.txt.* || true
rm -rf xla_dumps/scan-offload-ptxla || true
mkdir -p xla_dumps/scan-offload-ptxla

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2"

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export XLA_SAVE_TENSORS_FILE=ir_dumps/scan-offload-ptxla.txt
export XLA_SAVE_TENSORS_FMT=hlo
export XLA_FLAGS=--xla_dump_to=xla_dumps/scan-offload-ptxla

# Debugging notes:
# set print object on
# set print vtbl on
# b torch_xla/csrc/tensor.cpp:460
# set substitute-path torch_xla/csrc /workspaces/torch/pytorch/xla/torch_xla/csrc
# gdb --args python3 test_llm_offload.py --name ptxla-scan "$@"
python3 test_llm_offload.py --name ptxla-scan "$@"
