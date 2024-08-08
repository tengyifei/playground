from typing import Callable, TypeVar

import torch
import torch_xla.core.xla_builder as xb
from torch._ops import HigherOrderOperator
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented

import torch_xla

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


class ScanOp(HigherOrderOperator):

  def __init__(self):
    super().__init__("scan")

  def __call__(
      self,
      fn: Callable[[Carry, X], tuple[Carry, Y]],
      init: Carry,
      xs: X,
      /,
  ) -> tuple[Carry, Carry, Y]:
    return super().__call__(fn, init, xs)  # type: ignore


scan_op = ScanOp()


def _scan_carry_history(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
) -> tuple[Carry, Carry, Y]:
  return scan_op(fn, init, xs)


def scan(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
) -> tuple[Carry, Y]:
  carry, carry_history, ys = scan_op(fn, init, xs)
  return carry, ys


def dynamic_update_slice(ys: xb.Op, y: xb.Op, idx: xb.Op) -> xb.Op:
  # See https://openxla.org/xla/operation_semantics#dynamicupdateslice.
  y = y.broadcast([1])
  indices = [idx]
  for _ in range(ys.shape().rank - 1):  # TODO: crashes during backward
    indices.append(idx.zeros_like())
  return ys.dynamic_update_slice(y, indices)


def dynamic_slice(xs: xb.Op, idx: xb.Op) -> xb.Op:
  indices = [idx]
  for _ in range(xs.shape().rank - 1):
    indices.append(idx.zeros_like())
  slice_shape = list(xs.shape().sizes)
  slice_shape[0] = 1
  sliced = xs.dynamic_slice(indices, slice_shape)
  shape = list(xs.shape().sizes)
  shape = shape[1:]
  return sliced.reshape(shape)


# See https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md for
# the meaning of CompositeExplicitAutogradNonFunctional.
@scan_op.py_impl(DispatchKey.CompositeExplicitAutogradNonFunctional)
@scan_op.py_impl(DispatchKey.XLA)
def scan_dense(fn, init, xs):
  """Forward implementation of scan."""

  # Abstractly trace and lower `fn`.
  # Later we will include `fn_computation` within the while loop body.
  device = torch_xla.device()
  fake_carry = torch.empty(
      init.size(), dtype=init.dtype,
      requires_grad=init.requires_grad).to(device)
  fake_xs = torch.empty(
      xs[0].size(), dtype=xs[0].dtype,
      requires_grad=xs.requires_grad).to(device)
  fn_outputs = fn(fake_carry, fake_xs)
  fn_ctx = torch_xla._XLAC.lowering.LoweringContext()
  fn_ctx.set_name_string("my_ctx")
  fn_ctx.build(list(fn_outputs))
  fn_hlo = fn_ctx.hlo()
  fn_computation = xb.computation_from_module_proto("my_fn_computation", fn_hlo)
  xs_len = xs.shape[0]

  # Figure out the shape of `ys` from the abstract tracing.
  fn_carry_shape, fn_y_shape = (v.shape for v in fn_outputs)
  assert fn_carry_shape == init.shape, f"`fn` must keep the `carry` shape unchanged. \
    Got {fn_carry_shape} but expected {init.shape}"

  def cond_fn(num_iters: xb.Op, carry: xb.Op, xs: xb.Op,
              fn_carry_history: xb.Op, ys: xb.Op):
    return num_iters > xb.Op.scalar(num_iters.builder(), 0, dtype=xb.Type.S64)

  def body_fn(num_iters: xb.Op, carry: xb.Op, xs: xb.Op,
              fn_carry_history: xb.Op, ys: xb.Op):
    xs_len_op = xb.Op.scalar(num_iters.builder(), xs_len, dtype=xb.Type.S64)
    one = xb.Op.scalar(num_iters.builder(), 1, dtype=xb.Type.S64)
    idx = xs_len_op - num_iters
    x = dynamic_slice(xs, idx)
    fn_carry_history = dynamic_update_slice(fn_carry_history, carry, idx)
    result = xb.Op.call(fn_computation, (carry, x))
    carry = result.get_tuple_element(0)
    y = result.get_tuple_element(1)
    ys = dynamic_update_slice(ys, y, idx)
    return xb.Op.tuple((num_iters - one, carry, xs, fn_carry_history, ys))

  num_iters = torch.tensor(xs_len, device=device)
  ys = torch.zeros((xs_len, *fn_y_shape), device=device)
  fn_carry_history = torch.zeros((xs_len, *fn_carry_shape), device=device)
  carry = (num_iters, init, xs, fn_carry_history, ys)
  builder = xb.create_builder('scan')
  carry_param = []
  for i, val in enumerate(carry):
    carry_param.append(xb.mkparam(builder, i, xb.tensor_shape(val)))
  res = xb.Op.mkwhile(tuple(carry_param), cond_fn, body_fn)
  computation = res.build('scan')

  _last_iter, carry, xs, fn_carry_history, ys = torch_xla._XLAC._xla_user_computation(
      'xla::scan', carry, computation)

  return carry, fn_carry_history, ys


import torch.autograd
from torch.utils.checkpoint import detach_variable


class Scan(torch.autograd.Function):

  @staticmethod
  def forward(ctx, fn, init, xs):
    # Forward pass, save inputs for backward
    ctx._fn = fn
    with torch._C._AutoDispatchBelowAutograd():
      carry, carry_history, ys = _scan_carry_history(fn, init, xs)
    ctx.save_for_backward(carry_history, xs)
    return carry, carry_history, ys

  @staticmethod
  def backward(ctx, grad_carry, grad_carry_history, grad_ys):
    fn = ctx._fn
    carry_history, xs = ctx.saved_tensors

    def step_fn(grad_carry, grad_y, carry, x):
      # Compute the backward of a single scan iteration
      detached_inputs = detach_variable((carry, x))
      with torch.enable_grad():
        outputs = fn(*detached_inputs)
        torch.autograd.backward(outputs, (grad_carry, grad_y))
      grad_carry, grad_x = tuple(inp.grad for inp in detached_inputs)
      assert grad_carry is not None
      assert grad_x is not None
      return grad_carry, grad_x

    # Reverse loop to accumulate gradients
    grad_init = grad_carry.clone()
    grad_xs = torch.zeros_like(xs)

    xs_len = xs.size(0)
    carry_history = carry_history.flip(0).requires_grad_(True)
    xs = xs.flip(0).requires_grad_(True)
    grad_ys = grad_ys.flip(0).requires_grad_(True)
    for i in range(xs_len):
      grad_init, grad_xs[i] = step_fn(grad_init, grad_ys[i], carry_history[i],
                                      xs[i])

    return None, grad_init, grad_xs.flip(0)


scan_op.py_impl(DispatchKey.AutogradXLA)(Scan.apply)


@scan_op.py_functionalize_impl
def scan_func(ctx, fn, init, xs):
  unwrapped_init = ctx.unwrap_tensors(init)
  unwrapped_xs = ctx.unwrap_tensors(xs)
  with ctx.redispatch_to_next() as m:
    functional_fn = ctx.functionalize(fn)
    ret = scan_op(
        functional_fn,
        unwrapped_init,
        unwrapped_xs,
    )
    return ctx.wrap_tensors(ret)
