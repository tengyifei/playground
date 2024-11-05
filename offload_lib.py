import torch_xla
import torch_xla.runtime
from functorch.compile import aot_function
import torch
import itertools
import torch_xla.core.xla_model as xm
from torch_xla.debug.profiler import Trace
from torch_xla.experimental.stablehlo_custom_call import place_to_host, place_to_device


def offload(module: torch.nn.Module,
            use_sync_to_break_graph: bool = False) -> torch.nn.Module:
  from functorch.compile import aot_module

  def should_offload(t: torch.Tensor):
    for p in module.parameters():
      if t is p:
        print(f"Skip offloading {type(t)} {t.shape}")
        return False
    return True

  def maybe_place_to_host(t):
    if should_offload(t):
      print(f"Offload {t.shape} tensor to host")
      return place_to_host(t)
    else:
      return t

  def maybe_place_to_device(t):
    if should_offload(t):
      print(f"Bring back {t.shape} tensor to device")
      return place_to_device(t)
    else:
      return t

  # The compiler_fn is called after the forward and backward graphs are extracted.
  # Here, we just print the code in the compiler_fn. Return of this function is a callable.
  def forward_comp(fx_module: torch.fx.GraphModule, _):
    print("Forward", fx_module.code)

    def compute_then_offload(*args, **kwargs):
      for a in args:
        print("Arg type: ", type(a), str(a.shape))
      for k, v in kwargs.items():
        print(f"kwarg {k} type: ", type(v), str(v.shape))
      with Trace("fwd"):
        res = fx_module(*args, **kwargs)
        res2 = [res[0]] + [maybe_place_to_host(r) for r in res[1:]]
        xm.optimization_barrier_(res2)
        print("Forward output shapes: " +
              ', '.join(str(a.shape if a is not None else None) for a in res2))
        if use_sync_to_break_graph:
          torch_xla.sync()
        return res2

    return compute_then_offload

  def backward_comp(fx_module, _):
    print("Backward", fx_module.code)

    def compute_then_offload(*args, **kwargs):
      for a in args:
        print("Arg type: ", type(a), str(a.shape))
      for k, v in kwargs.items():
        print(f"kwarg {k} type: ", type(v), str(v.shape))
      with Trace("bwd"):
        # This barrier matches what we got from a simple JAX example.
        xm.optimization_barrier_(list(itertools.chain(args, kwargs.values())))
        args = [maybe_place_to_device(r) for r in args]
        kwargs = {k: maybe_place_to_device(v) for k, v in kwargs.items()}
        res = fx_module(*args, **kwargs)
        print("Backward output shapes: " +
              ', '.join(str(a.shape if a is not None else None) for a in res))
        if use_sync_to_break_graph:
          torch_xla.sync()
        return res

    return compute_then_offload

  # Pass on the compiler_fn to the aot_module API
  return aot_module(module, fw_compiler=forward_comp, bw_compiler=backward_comp)
