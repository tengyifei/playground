#!/usr/bin/env python3
"""Test JAX AOT cross-compilation for TPU using MaxText's approach"""

import jax
import jax.numpy as jnp
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh
from jax.experimental.pallas.tpu import get_tpu_info

def test_cross_compile_for_tpu():
    """
    Test AOT compilation targeting TPU hardware using virtual topology.
    This follows MaxText's approach in train_compile.py
    """

    print("=== JAX AOT Cross-Compilation for TPU ===\n")
    print(f"JAX version: {jax.__version__}")
    print(f"Current devices: {jax.devices()}")
    print(f"Current backend: {jax.default_backend()}\n")

    # Define target TPU topology
    # Format: "chip_version=dimensions" or "chip_version:dimensions"
    # v5e uses 2D: "v5e=2x2", "v5e=4x4", "v5e=8x16"
    # v4 uses 3D: "v4=2x2x1", "v4=4x4x8"
    target_platform = "tpu"
    target_topology = "v5e=2x2"  # v5e-4 configuration (2x2 2D mesh)

    print(f"Target Platform: {target_platform}")
    print(f"Target Topology: {target_topology}\n")

    try:
        # Create virtual topology for the target TPU
        # This creates a "mock" device topology without needing actual TPU hardware
        topology_desc = get_topology_desc(
            topology_name=target_topology,
            platform=target_platform,
        )

        print(f"✓ Created virtual topology with {len(topology_desc.devices)} devices")
        print(f"  First device: {topology_desc.devices[0]}")
        if len(topology_desc.devices) > 1:
            print(f"  Last device: {topology_desc.devices[-1]}")

        # Now get TPU info for this topology
        # Note: This requires setting JAX's default backend to use these devices
        with jax.default_device(topology_desc.devices[0]):
            try:
                tpu_info = get_tpu_info()
                print("\n=== Target TPU Hardware Information (per TensorCore) ===")
                print(f"Chip Version: {tpu_info.chip_version}")
                print(f"Generation: TPU v{tpu_info.generation}")
                print(f"Number of Cores: {tpu_info.num_cores}")
                print(f"Number of Lanes: {tpu_info.num_lanes}")
                print(f"Number of Sublanes: {tpu_info.num_sublanes}")
                print(f"MXU Column Size: {tpu_info.mxu_column_size}")
                print(f"VMEM Capacity: {tpu_info.vmem_capacity_bytes / (1024**2):.2f} MB")
                print(f"CMEM Capacity: {tpu_info.cmem_capacity_bytes / (1024**2):.2f} MB")
                print(f"SMEM Capacity: {tpu_info.smem_capacity_bytes / (1024**2):.2f} MB")
                print(f"HBM Capacity: {tpu_info.hbm_capacity_bytes / (1024**3):.2f} GB")
                print(f"Memory Bandwidth: {tpu_info.mem_bw_bytes_per_second / (1024**3):.2f} GB/s")
                print(f"BF16 FLOPS: {tpu_info.bf16_ops_per_second / 1e12:.2f} TFLOPS")
                print(f"INT8 OPS: {tpu_info.int8_ops_per_second / 1e12:.2f} TOPS")
                print(f"FP8 FLOPS: {tpu_info.fp8_ops_per_second / 1e12:.2f} TFLOPS")
                print(f"INT4 OPS: {tpu_info.int4_ops_per_second / 1e12:.2f} TOPS")
                print(f"Is Lite: {tpu_info.is_lite}")
                print(f"Is Split Chip: {tpu_info.is_split_chip}")
            except Exception as e:
                print(f"\n⚠ Could not get TPU info from virtual topology: {e}")

        # Create a simple function to compile
        def matmul_add(x, y, z):
            """Simple matrix multiply and add"""
            return jnp.matmul(x, y) + z

        # Create sample inputs with shapes
        batch_size, seq_len, hidden_dim = 4, 128, 512
        x = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.bfloat16)
        y = jnp.ones((batch_size, hidden_dim, hidden_dim), dtype=jnp.bfloat16)
        z = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.bfloat16)

        print(f"\n=== Compiling function for target topology ===")
        print(f"Input shapes: x={x.shape}, y={y.shape}, z={z.shape}")
        print(f"Input dtype: {x.dtype}\n")

        # AOT compilation workflow
        # Step 1: JIT the function
        jitted_fn = jax.jit(matmul_add)

        # Step 2: Lower to StableHLO targeting the virtual TPU devices
        # By using the topology devices, the lowering will target TPU
        with jax.default_device(topology_desc.devices[0]):
            lowered = jitted_fn.lower(x, y, z)
            print("✓ Successfully lowered function to StableHLO")

            # Step 3: Compile the lowered function
            compiled = lowered.compile()
            print("✓ Successfully compiled function for TPU")

            # Analyze the compiled executable
            try:
                memory_stats = compiled.memory_analysis()
                print("\n=== Compiled Memory Analysis ===")
                if hasattr(memory_stats, 'temp_size_in_bytes'):
                    print(f"Temp memory: {memory_stats.temp_size_in_bytes / (1024**2):.2f} MB")
                if hasattr(memory_stats, 'output_size_in_bytes'):
                    print(f"Output memory: {memory_stats.output_size_in_bytes / (1024**2):.2f} MB")
            except Exception as e:
                print(f"Memory analysis not available: {e}")

            # Print the HLO IR
            print("\n=== StableHLO IR (first 1000 chars) ===")
            hlo_text = lowered.as_text()
            print(hlo_text[:1000])
            if len(hlo_text) > 1000:
                print("...")

            # The compiled executable can be serialized for later use
            # from jax.experimental.serialize_executable import serialize
            # serialized = serialize(compiled)
            # with open("compiled_fn.bin", "wb") as f:
            #     f.write(serialized)

            print("\n✓ AOT cross-compilation successful!")
            print("  The compiled executable is ready for execution on actual TPU hardware")

    except Exception as e:
        print(f"\n✗ Error during cross-compilation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cross_compile_for_tpu()
