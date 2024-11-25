import torch
import torch_xla
import torch_xla.utils.checkpoint
import torch_xla.core.xla_model as xm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.weak import WeakTensorKeyDictionary
import contextlib
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map_only
from torch.utils.checkpoint import checkpoint


class MarkInputsToRegion(TorchFunctionMode):

  def __init__(self, mark_fn):
    # tensor -> bool
    self.is_marked = WeakTensorKeyDictionary()
    self.mark_fn = mark_fn

  # This will be called on every torch function call during backwards.
  def __torch_function__(self, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}

    def mark(x):
      self.mark_fn(x)
      self.is_marked[x] = True

    print("torch_function", func, types, args)
    tree_map_only(torch.Tensor, mark, (args, kwargs))
    out = func(*args, **kwargs)
    tree_map_only(torch.Tensor, mark, out)
    return out


def context_fn():

  def mark_fn(x):
    print("input to region: ", x)

  # First context wraps initial computation.
  # Second context wraps recomputation.
  return contextlib.nullcontext(), MarkInputsToRegion(mark_fn)


# Test a tensor that is closed over
y = torch.tensor([2.], requires_grad=True)
x = torch.tensor([1.], requires_grad=True)


def func(x):
  # the output of this mul or this clone should not be wrapped
  out = (x + 123) * y
  return out.clone()


out = checkpoint(func, x, context_fn=context_fn, use_reentrant=False)
assert out is not None
out.sum().backward()
