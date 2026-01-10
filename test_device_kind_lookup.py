#!/usr/bin/env python3
"""Demonstrate how get_tpu_info() looks up device information

This script shows the internal mechanism of how JAX's get_tpu_info() function
determines which TPU generation specs to return based on the default device context.
"""

import jax
from jax.experimental.topologies import get_topology_desc
from jax.experimental.pallas.tpu import get_tpu_info
from jax.extend import backend as jex_backend

def demonstrate_device_lookup():
    """Show how device_kind is determined at each step"""

    print("="*80)
    print("HOW get_tpu_info() LOOKS UP TPU INFORMATION")
    print("="*80)
    print()

    # Step 1: Create a virtual TPU topology
    print("STEP 1: Create virtual TPU topology")
    print("-" * 80)
    topology_name = "v5e=2x2"
    topology_desc = get_topology_desc(topology_name=topology_name, platform="tpu")
    device = topology_desc.devices[0]

    print(f"Topology name: {topology_name}")
    print(f"Number of devices: {len(topology_desc.devices)}")
    print(f"First device: {device}")
    print(f"Device type: {type(device)}")
    print()

    # Step 2: Check device properties
    print("STEP 2: Inspect device object properties")
    print("-" * 80)
    print(f"device.device_kind: '{device.device_kind}'")
    print(f"device.platform: '{device.platform}'")
    if hasattr(device, 'num_cores'):
        print(f"device.num_cores: {device.num_cores}")
    print()

    # Step 3: Show default device before context
    print("STEP 3: Check default device BEFORE jax.default_device()")
    print("-" * 80)
    default_before = jex_backend.get_default_device()
    print(f"Default device: {default_before}")
    print(f"Default device kind: '{default_before.device_kind}'")
    print()

    # Step 4: Set default device and check
    print("STEP 4: Set default device with jax.default_device() context")
    print("-" * 80)
    print(f"Entering context: jax.default_device({device})")

    with jax.default_device(device):
        default_in_context = jex_backend.get_default_device()
        print(f"Default device in context: {default_in_context}")
        print(f"Default device kind in context: '{default_in_context.device_kind}'")
        print()

        # Step 5: Show how get_tpu_info uses this
        print("STEP 5: Call get_tpu_info() to see it use the device_kind")
        print("-" * 80)

        # Manually show what get_tpu_info does internally
        from jax._src.pallas.mosaic import core as mosaic_core
        device_kind_str = mosaic_core.get_device_kind()
        print(f"mosaic_core.get_device_kind() returns: '{device_kind_str}'")
        print()

        print("This string is matched in get_tpu_info():")
        print(f"  case '{device_kind_str}' | 'TPU v5e':")
        print(f"      return TpuInfo(chip_version=ChipVersion.TPU_V5E, generation=5, ...)")
        print()

        # Actually call get_tpu_info
        tpu_info = get_tpu_info()
        print(f"Result from get_tpu_info():")
        print(f"  Chip Version: {tpu_info.chip_version}")
        print(f"  Generation: {tpu_info.generation}")
        print(f"  BF16 Performance: {tpu_info.bf16_ops_per_second / 1e12:.2f} TFLOPS")
        print()

    # Step 6: Check after context exits
    print("STEP 6: Check default device AFTER context exits")
    print("-" * 80)
    default_after = jex_backend.get_default_device()
    print(f"Default device: {default_after}")
    print(f"Default device kind: '{default_after.device_kind}'")
    print()

    # Step 7: Demonstrate with different topology
    print("="*80)
    print("STEP 7: Try with different TPU generation (v6e)")
    print("="*80)
    print()

    topology_v6e = get_topology_desc(topology_name="v6e=2x2", platform="tpu")
    device_v6e = topology_v6e.devices[0]

    print(f"Device: {device_v6e}")
    print(f"Device kind: '{device_v6e.device_kind}'")

    with jax.default_device(device_v6e):
        from jax._src.pallas.mosaic import core as mosaic_core
        device_kind_str = mosaic_core.get_device_kind()
        print(f"get_device_kind() returns: '{device_kind_str}'")

        tpu_info = get_tpu_info()
        print(f"\nResult:")
        print(f"  Chip Version: {tpu_info.chip_version}")
        print(f"  Generation: {tpu_info.generation}")
        print(f"  BF16 Performance: {tpu_info.bf16_ops_per_second / 1e12:.2f} TFLOPS")
        print(f"  FP8 Performance: {tpu_info.fp8_ops_per_second / 1e12:.2f} TFLOPS")
    print()

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("The lookup chain:")
    print("  1. jax.default_device(device) → sets config.default_device.value")
    print("  2. get_tpu_info() → calls core.get_device_kind()")
    print("  3. get_device_kind() → calls get_default_device().device_kind")
    print("  4. get_default_device() → returns config.default_device.value")
    print("  5. device.device_kind → returns string like 'TPU v5 lite'")
    print("  6. get_tpu_info() → matches string and returns hardcoded specs")
    print()
    print("This allows querying any TPU generation without physical hardware!")
    print()


if __name__ == "__main__":
    demonstrate_device_lookup()
