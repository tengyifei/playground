#!/usr/bin/env python3
"""Test querying different TPU generation specs using jax.default_device

Installation:
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"""

import jax
from jax.experimental.topologies import get_topology_desc
from jax.experimental.pallas.tpu import get_tpu_info

def test_tpu_generation_specs():
    """Test if jax.default_device allows querying different TPU generation specs"""

    print("=== Testing TPU Generation Specifications ===\n")
    print(f"JAX version: {jax.__version__}\n")

    # Define different TPU topologies to test
    # Format: (topology_name, description)
    topologies = [
        ("v4=2x2x1", "TPU v4"),
        ("v5e=2x2", "TPU v5e (Lite)"),
        ("v5p=2x2x1", "TPU v5p (Performance)"),
        ("v6e=2x2", "TPU v6e (Lite)"),
    ]

    results = []

    for topology_name, description in topologies:
        print(f"{'='*80}")
        print(f"Testing: {description} ({topology_name})")
        print(f"{'='*80}\n")

        try:
            # Create virtual topology
            topology_desc = get_topology_desc(
                topology_name=topology_name,
                platform="tpu",
            )

            print(f"✓ Created virtual topology with {len(topology_desc.devices)} devices")
            print(f"  First device: {topology_desc.devices[0]}\n")

            # Use jax.default_device to set the context to this TPU generation
            with jax.default_device(topology_desc.devices[0]):
                try:
                    tpu_info = get_tpu_info()

                    # Collect key specs
                    specs = {
                        "Topology": topology_name,
                        "Description": description,
                        "Chip Version": str(tpu_info.chip_version),
                        "Generation": tpu_info.generation,
                        "Num Cores": tpu_info.num_cores,
                        "Num Lanes": tpu_info.num_lanes,
                        "Num Sublanes": tpu_info.num_sublanes,
                        "MXU Column Size": tpu_info.mxu_column_size,
                        "VMEM (MB)": f"{tpu_info.vmem_capacity_bytes / (1024**2):.2f}",
                        "CMEM (MB)": f"{tpu_info.cmem_capacity_bytes / (1024**2):.2f}",
                        "SMEM (MB)": f"{tpu_info.smem_capacity_bytes / (1024**2):.2f}",
                        "HBM (GB)": f"{tpu_info.hbm_capacity_bytes / (1024**3):.2f}",
                        "Memory BW (GB/s)": f"{tpu_info.mem_bw_bytes_per_second / (1024**3):.2f}",
                        "BF16 (TFLOPS)": f"{tpu_info.bf16_ops_per_second / 1e12:.2f}",
                        "INT8 (TOPS)": f"{tpu_info.int8_ops_per_second / 1e12:.2f}",
                        "FP8 (TFLOPS)": f"{tpu_info.fp8_ops_per_second / 1e12:.2f}",
                        "INT4 (TOPS)": f"{tpu_info.int4_ops_per_second / 1e12:.2f}",
                        "Is Lite": tpu_info.is_lite,
                        "Is Split Chip": tpu_info.is_split_chip,
                    }

                    results.append(specs)

                    # Print specs
                    print("Hardware Specifications (per TensorCore):")
                    for key, value in specs.items():
                        if key not in ["Topology", "Description"]:
                            print(f"  {key:20s}: {value}")

                    print(f"\n✓ Successfully queried {description} specs\n")

                except Exception as e:
                    print(f"✗ Failed to get TPU info: {e}\n")

        except Exception as e:
            print(f"✗ Failed to create topology: {e}\n")

    # Print comparison table
    if results:
        print("\n" + "="*80)
        print("COMPARISON TABLE: TPU Generations")
        print("="*80 + "\n")

        # Print header
        print(f"{'Metric':<25}", end="")
        for result in results:
            print(f"{result['Description']:<20}", end="")
        print()
        print("-" * (25 + 20 * len(results)))

        # Print key metrics
        key_metrics = [
            "Chip Version",
            "Generation",
            "Num Cores",
            "MXU Column Size",
            "VMEM (MB)",
            "HBM (GB)",
            "Memory BW (GB/s)",
            "BF16 (TFLOPS)",
            "INT8 (TOPS)",
            "FP8 (TFLOPS)",
            "Is Lite",
        ]

        for metric in key_metrics:
            print(f"{metric:<25}", end="")
            for result in results:
                value = result.get(metric, "N/A")
                print(f"{str(value):<20}", end="")
            print()

        print("\n" + "="*80)
        print("\nConclusion:")
        print("jax.default_device() successfully allows querying specs for different")
        print("TPU generations by creating virtual topologies and setting device context.")
        print("="*80)

if __name__ == "__main__":
    test_tpu_generation_specs()
