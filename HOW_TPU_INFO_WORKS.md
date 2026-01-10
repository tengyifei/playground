# How `get_tpu_info()` Looks Up TPU Information

## Summary

`get_tpu_info()` uses JAX's device context to determine which TPU generation to query. When you use `jax.default_device()`, it sets the default device in JAX's config, which `get_tpu_info()` then reads to determine the device kind.

## Call Chain

```
get_tpu_info()
    ↓
core.get_device_kind()
    ↓
jax.sharding.get_abstract_mesh().abstract_device.device_kind
    (if in a mesh context)
    OR
    ↓
jex_backend.get_default_device().device_kind
    ↓
config.default_device.value.device_kind
    (returns the device set by jax.default_device())
```

## Source Code Analysis

### 1. `get_tpu_info()` function
Location: `/usr/local/lib/python3.11/dist-packages/jax/_src/pallas/mosaic/tpu_info.py`

```python
@jax_util.cache(trace_context_in_key=True)
def get_tpu_info() -> TpuInfo:
  """Returns the TPU hardware information for the current device."""
  device_kind = core.get_device_kind()  # ← Gets device kind string

  match device_kind:
    case "TPU v5 lite" | "TPU v5e":
      return TpuInfo(
          chip_version=ChipVersion.TPU_V5E,
          generation=5,
          ...
      )
    case "TPU v4":
      return TpuInfo(...)
    # ... other cases
```

The key is `core.get_device_kind()` which retrieves a string like `"TPU v5 lite"`, `"TPU v4"`, etc.

### 2. `get_device_kind()` function
Location: `/usr/local/lib/python3.11/dist-packages/jax/_src/pallas/mosaic/core.py`

```python
def get_device_kind() -> str:
  # First check if we're in a mesh context
  if abstract_device := jax.sharding.get_abstract_mesh().abstract_device:
    return abstract_device.device_kind

  # Otherwise use the default device
  return jex_backend.get_default_device().device_kind
```

This function has two paths:
- **Mesh context**: Uses abstract device from the current mesh
- **Default**: Uses JAX's default device

### 3. `get_default_device()` function
Location: `jax.extend.backend`

```python
def get_default_device() -> xc.Device:
  if isinstance(config.default_device.value, str):
    return xb.get_backend(config.default_device.value).local_devices()[0]
  else:
    return config.default_device.value or xb.local_devices()[0]
```

This reads from `config.default_device.value`, which is set by `jax.default_device()`.

### 4. `jax.default_device()` context manager
Location: `jax._src.config`

```python
default_device = string_or_object_state(
    name='jax_default_device',
    default=None,
    help='Configure the default device for JAX operations...'
)
```

This is a config state that can be set to a Device object.

## How Virtual Topologies Work

When you create a virtual topology:

```python
from jax.experimental.topologies import get_topology_desc

# Create virtual TPU v5e topology
topology_desc = get_topology_desc(
    topology_name="v5e=2x2",  # 2x2 mesh = 4 devices
    platform="tpu",
)

# The devices have a device_kind property
device = topology_desc.devices[0]
print(device.device_kind)  # Output: "TPU v5 lite"
```

The virtual devices are created with the correct `device_kind` property set based on the topology name.

## Using `jax.default_device()` to Query Different Generations

```python
import jax
from jax.experimental.topologies import get_topology_desc
from jax.experimental.pallas.tpu import get_tpu_info

# Create virtual v5e topology
topology_desc = get_topology_desc(topology_name="v5e=2x2", platform="tpu")

# Set default device to the first virtual TPU
with jax.default_device(topology_desc.devices[0]):
    # Now get_tpu_info() queries this virtual device
    tpu_info = get_tpu_info()
    print(f"Generation: {tpu_info.generation}")  # Output: 5
    print(f"Chip: {tpu_info.chip_version}")      # Output: v5e
```

### What happens:
1. `jax.default_device(device)` sets `config.default_device.value = device`
2. `get_tpu_info()` calls `core.get_device_kind()`
3. `get_device_kind()` calls `get_default_device()`
4. `get_default_device()` returns the device we set
5. `device.device_kind` returns `"TPU v5 lite"`
6. `get_tpu_info()` matches this string and returns v5e specs

## Device Kind String Mapping

The device kind strings map to TPU generations:

| Topology Name | Device Kind String | Chip Version | Generation |
|--------------|-------------------|--------------|-----------|
| `v4=2x2x1`   | `"TPU v4"`        | TPU_V4       | 4         |
| `v5e=2x2`    | `"TPU v5 lite"`   | TPU_V5E      | 5         |
| `v5p=2x2x1`  | `"TPU v5p"`       | TPU_V5P      | 5         |
| `v6e=2x2`    | `"TPU v6e"`       | TPU_V6E      | 6         |

## Key Insights

1. **No actual hardware needed**: Virtual topologies create device objects with the correct `device_kind` property
2. **Context-based**: `jax.default_device()` temporarily sets which device is "current"
3. **Hardcoded specs**: All TPU specs are hardcoded in `tpu_info.py` based on device kind string
4. **Caching**: Results are cached per trace context for performance

## Example: Querying Multiple Generations

```python
topologies = [
    ("v4=2x2x1", "TPU v4"),
    ("v5e=2x2", "TPU v5e"),
    ("v6e=2x2", "TPU v6e"),
]

for topology_name, desc in topologies:
    topology_desc = get_topology_desc(topology_name=topology_name, platform="tpu")

    with jax.default_device(topology_desc.devices[0]):
        info = get_tpu_info()
        print(f"{desc}: {info.bf16_ops_per_second / 1e12:.2f} TFLOPS")

# Output:
# TPU v4: 137.50 TFLOPS
# TPU v5e: 197.00 TFLOPS
# TPU v6e: 920.00 TFLOPS
```

This allows you to query specifications for any TPU generation without physical hardware!
