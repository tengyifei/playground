#!/usr/bin/env python3
"""Test JAX AOT compilation for TPU"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas.tpu import get_tpu_info

def test_aot_compile():
    """Test AOT compilation with jax.jit().lower().compile()"""

    # Simple function to test
    def add_matrices(x, y):
        return x + y

    # Create sample inputs
    x = jnp.ones((10, 10))
    y = jnp.ones((10, 10))

    print("Testing JAX AOT compilation for TPU...")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Get TPU info
    try:
        tpu_info = get_tpu_info()
        print("\n=== TPU Hardware Information (per TensorCore) ===")
        print(f"Chip Version: {tpu_info.chip_version}")
        print(f"Generation: {tpu_info.generation}")
        print(f"Number of Cores: {tpu_info.num_cores}")
        print(f"Number of Lanes: {tpu_info.num_lanes}")
        print(f"Number of Sublanes: {tpu_info.num_sublanes}")
        print(f"MXU Column Size: {tpu_info.mxu_column_size}")
        print(f"VMEM Capacity: {tpu_info.vmem_capacity_bytes / (1024**2):.2f} MB")
        print(f"CMEM Capacity: {tpu_info.cmem_capacity_bytes / (1024**2):.2f} MB")
        print(f"SMEM Capacity: {tpu_info.smem_capacity_bytes / (1024**2):.2f} MB")
        print(f"HBM Capacity: {tpu_info.hbm_capacity_bytes / (1024**3):.2f} GB")
        print(f"Memory Bandwidth: {tpu_info.mem_bw_bytes_per_second / (1024**3):.2f} GB/s")
        print(f"BF16 Ops/s: {tpu_info.bf16_ops_per_second / 1e12:.2f} TFLOPS")
        print(f"INT8 Ops/s: {tpu_info.int8_ops_per_second / 1e12:.2f} TOPS")
        print(f"FP8 Ops/s: {tpu_info.fp8_ops_per_second / 1e12:.2f} TFLOPS")
        print(f"INT4 Ops/s: {tpu_info.int4_ops_per_second / 1e12:.2f} TOPS")
        print(f"Is Lite: {tpu_info.is_lite}")
        print(f"Is Split Chip: {tpu_info.is_split_chip}")
        if tpu_info.sparse_core:
            print(f"Sparse Core: {tpu_info.sparse_core}")
    except Exception as e:
        print(f"\n⚠ Could not get TPU info: {e}")

    # Test AOT compilation
    try:
        # Lower the function
        lowered = jax.jit(add_matrices).lower(x, y)
        print("\n✓ Successfully lowered function")

        # Compile the lowered function
        compiled = lowered.compile()
        print("✓ Successfully compiled function")

        # Run the compiled function
        result = compiled(x, y)
        print(f"✓ Successfully executed compiled function")
        print(f"Result shape: {result.shape}")
        print(f"Result sum: {result.sum()}")

        # Print HLO if available
        print("\nHLO IR:")
        print(lowered.as_text())

    except Exception as e:
        print(f"✗ Error during AOT compilation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_aot_compile()
