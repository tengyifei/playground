from typing import Callable, Sequence, TypeVar, Tuple

import torch
import torch_xla.core.xla_builder as xb
from torch._ops import HigherOrderOperator
from torch._C import DispatchKey
from torch.utils._pytree import tree_map, PyTree, tree_flatten, tree_unflatten

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

  flat_init, carry_spec = tree_flatten(init)
  _, xs_spec = tree_flatten(xs)

  # Because `flat_fn` returns a concatenated flattened carry and y list,
  # we need to know how many elements out of that list is the carry.
  flat_carry_len = len(flat_init)

  # `fn` operates on PyTrees and returns PyTrees. However, XLA only understands
  # (lists or tuples of) tensors. So we will craft a `flat_fn` that takes in
  # flattened PyTrees, internally recreates the desired tree structure, then
  # calls `fn`. Similar transformation on the return path.
  def flat_fn(carry: Sequence[torch.Tensor],
              x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    carry_pytree = tree_unflatten(carry, carry_spec)
    x_pytree = tree_unflatten(x, xs_spec)
    carry_pytree, y_pytree = fn(carry_pytree, x_pytree)
    carry, _ = tree_flatten(carry_pytree)
    y, _ = tree_flatten(y_pytree)
    return carry + y

  # Abstractly trace and lower `fn`.
  # Later we will include `fn_computation` within the while loop body.
  device = torch_xla.device()
  fake_carry_pytree = tree_map(
      lambda v: torch.empty(
          v.size(), dtype=v.dtype, requires_grad=v.requires_grad).to(device),
      init)
  fake_x_pytree = tree_map(
      lambda v: torch.empty(
          v[0].size(), dtype=v[0].dtype, requires_grad=v.requires_grad).to(
              device), xs)
  fn_output_carry_pytree, fn_output_y_pytree = fn(fake_carry_pytree,
                                                  fake_x_pytree)
  # Later we'll use `fn_output_carry_spec` etc to turn flattened outputs back to a PyTree.
  _, fn_output_carry_spec = tree_flatten(fn_output_carry_pytree)
  _, fn_output_y_spec = tree_flatten(fn_output_y_pytree)
  fake_carry, _ = tree_flatten(fake_carry_pytree)
  fake_x, _ = tree_flatten(fake_x_pytree)
  fn_outputs = flat_fn(fake_carry, fake_x)
  fn_ctx = torch_xla._XLAC.lowering.LoweringContext()
  fn_ctx.set_name_string("my_ctx")
  fn_ctx.build(list(fn_outputs))
  fn_hlo = fn_ctx.hlo()
  fn_computation = xb.computation_from_module_proto("my_fn_computation", fn_hlo)
  xs_len = fake_x[0].shape[0]

  # Since we are threading three PyTrees through the body_fn:
  # - carry: the scan state
  # - fn_carry_history: history of that state
  # - ys: the output of fn
  #
  # We need to concatenate all three into one big list prior to
  # entering `body_fn` and `cond_fn`, and split them back to three
  # objects which is easier to work with after that. This pair of
  # functions is for that purpose.
  def pack(carry: Sequence[xb.Op], fn_carry_history: Sequence[xb.Op],
           ys: Sequence[xb.Op]) -> Sequence[xb.Op]:
    return tuple(carry) + tuple(fn_carry_history) + tuple(ys)

  def unpack(
      seq: Sequence[xb.Op]
  ) -> Tuple[Sequence[xb.Op], Sequence[xb.Op], Sequence[xb.Op]]:
    seq = list(seq)
    carry = seq[:flat_carry_len]
    fn_carry_history = seq[flat_carry_len:flat_carry_len * 2]
    ys = seq[flat_carry_len * 2:]
    return carry, fn_carry_history, ys

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

  # Unflatten tensors back to PyTrees
  carry = [carry]
  ys = [ys]
  return tree_unflatten(carry, carry_spec), fn_carry_history, tree_unflatten(
      ys, fn_output_y_spec)


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


if __name__ == "__main__":
  device = torch_xla.device()

  # A simple function to be applied at each step of the scan
  def step_fn(carry, x):
    new_carry = carry + x
    y = carry * x
    return new_carry, y

  # Initial carry (let's make it a scalar with requires_grad)
  init_carry = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, device=device)

  # Example input tensor of shape (batch_size, features)
  xs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    requires_grad=True,
                    device=device)

  # Use the scan function
  final_carry, ys = scan(step_fn, init_carry, xs)

  # Loss for backward pass (sum of the outputs)
  loss = ys.sum()
  print(loss)
  assert loss.item() == 249.0

  loss.backward()
  torch_xla.sync()

  print("init_carry grad", init_carry.grad)
  print("xs grad", xs.grad)

  assert init_carry.grad is not None
  assert xs.grad is not None

  assert np.allclose(init_carry.grad.detach().cpu().numpy(),
                     np.array([12., 15., 18.]))

  assert np.allclose(xs.grad.detach().cpu().numpy(),
                     np.array([[12., 14., 16.], [9., 11., 13.], [6., 8., 10.]]))
