#!/bin/bash

# Dependencies:
# pip install jax optax flax 'jax[tpu]' tensorboardX tensorflow-cpu tensorboard-plugin-profile -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Initialize an associative array to store exit codes
declare -A exit_codes

# Function to run a command and store its exit code
run_and_store() {
    mkdir -p logs
    mkdir -p "xla_dumps/${name}"
    local name="$1"
    shift
    export XLA_FLAGS="--xla_dump_to=xla_dumps/${name}"
    echo "Running: $@"
    "$@" >> "logs/${name}.log" 2>&1
    local exit_code=$?
    exit_codes["$name"]=$exit_code
    export XLA_FLAGS=""
    echo "Completed: $name with exit code $exit_code"
}

# Debug build (optional)
export TPU_LIBRARY_PATH=/workspaces/torch/_libtpu.so

export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-0' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-1' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-2' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-3' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-4' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-5' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-6' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-7' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-all-reduce' python3 minimal-all-reduce-offload-crash-repro.py

export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2'
run_and_store 'skip-all-reduce-all-gather' python3 minimal-all-reduce-offload-crash-repro.py

# After all commands have run, print the exit codes
echo -e "\nSummary of Exit Codes:"
for name in "${!exit_codes[@]}"; do
    echo "${name}: ${exit_codes[$name]}"
done
