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

# Set initial LIBTPU_INIT_ARGS
export LIBTPU_INIT_ARGS='--xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_max_concurrent_host_copy=50 xla_latency_hiding_scheduler_rerun=0 --xla_jf_rematerialization_percent_shared_memory_limit=100'

# Run the first set of Python commands
run_and_store 'vanilla-30' python3 test_jax_llm_offload.py --name 'vanilla-30' --num-layers 30
run_and_store 'vanilla-40' python3 test_jax_llm_offload.py --name 'vanilla-40' --num-layers 40

run_and_store 'offload-30' python3 test_jax_llm_offload.py --name 'offload-30' --num-layers 30 --offload
run_and_store 'offload-40' python3 test_jax_llm_offload.py --name 'offload-40' --num-layers 40 --offload

run_and_store 'scan-offload-30' python3 test_jax_llm_offload.py --name 'scan-offload-30' --num-layers 30 --offload --scan
run_and_store 'scan-offload-40' python3 test_jax_llm_offload.py --name 'scan-offload-40' --num-layers 40 --offload --scan

# Update LIBTPU_INIT_ARGS for low memory settings
export LIBTPU_INIT_ARGS='--xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_max_concurrent_host_copy=6 xla_latency_hiding_scheduler_rerun=0 --xla_jf_rematerialization_percent_shared_memory_limit=50'

# Run the second set of Python commands with low memory settings
run_and_store 'vanilla-30-low-mem' python3 test_jax_llm_offload.py --name 'vanilla-30-low-mem' --num-layers 30
run_and_store 'vanilla-40-low-mem' python3 test_jax_llm_offload.py --name 'vanilla-40-low-mem' --num-layers 40

run_and_store 'offload-30-low-mem' python3 test_jax_llm_offload.py --name 'offload-30-low-mem' --num-layers 30 --offload
run_and_store 'offload-40-low-mem' python3 test_jax_llm_offload.py --name 'offload-40-low-mem' --num-layers 40 --offload

run_and_store 'scan-offload-30-low-mem' python3 test_jax_llm_offload.py --name 'scan-offload-30-low-mem' --num-layers 30 --offload --scan
run_and_store 'scan-offload-40-low-mem' python3 test_jax_llm_offload.py --name 'scan-offload-40-low-mem' --num-layers 40 --offload --scan

# Update LIBTPU_INIT_ARGS for even lower memory settings, and activate barrier.
export LIBTPU_INIT_ARGS='--xla_tpu_aggressive_opt_barrier_removal=DISABLED --xla_max_concurrent_host_copy=1 xla_latency_hiding_scheduler_rerun=0 --xla_jf_rematerialization_percent_shared_memory_limit=50'

# Run the second set of Python commands with low memory settings
run_and_store 'vanilla-30-lowest-mem' python3 test_jax_llm_offload.py --name 'vanilla-30-lowest-mem' --num-layers 30
run_and_store 'vanilla-40-lowest-mem' python3 test_jax_llm_offload.py --name 'vanilla-40-lowest-mem' --num-layers 40

run_and_store 'offload-30-lowest-mem' python3 test_jax_llm_offload.py --name 'offload-30-lowest-mem' --num-layers 30 --offload
run_and_store 'offload-40-lowest-mem' python3 test_jax_llm_offload.py --name 'offload-40-lowest-mem' --num-layers 40 --offload

run_and_store 'scan-offload-30-lowest-mem' python3 test_jax_llm_offload.py --name 'scan-offload-30-lowest-mem' --num-layers 30 --offload --scan
run_and_store 'scan-offload-40-lowest-mem' python3 test_jax_llm_offload.py --name 'scan-offload-40-lowest-mem' --num-layers 40 --offload --scan


# After all commands have run, print the exit codes
echo -e "\nSummary of Exit Codes:"
for name in "${!exit_codes[@]}"; do
    echo "${name}: ${exit_codes[$name]}"
done
