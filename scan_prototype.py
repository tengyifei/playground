import itertools
from typing import Callable, Dict, Sequence, TypeVar, Tuple, List, Any

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
  for _ in range(ys.shape().rank - 1):
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


class Builder:

  def __init__(self, name: str):
    self._builder = xb.create_builder(name)
    self._params = []
    self._param_tensors = []

  def add_param(self, val: torch.Tensor):
    idx = len(self._params)
    param = xb.mkparam(self._builder, idx, xb.tensor_shape(val))
    self._params.append(param)
    self._param_tensors.append(val)
    return idx

  def params(self) -> Tuple[xb.Op, ...]:
    return tuple(self._params)

  def param_tensors(self) -> Tuple[torch.Tensor, ...]:
    return tuple(self._param_tensors)

  def num_params(self) -> int:
    return len(self._params)


# See https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md for
# the meaning of CompositeExplicitAutogradNonFunctional.
@scan_op.py_impl(DispatchKey.CompositeExplicitAutogradNonFunctional)
@scan_op.py_impl(DispatchKey.XLA)
def scan_dense(fn, init, xs):
  """Forward implementation of scan."""

  flat_init, carry_spec = tree_flatten(init)
  flat_xs, xs_spec = tree_flatten(xs)

  # Because `flat_fn` returns a concatenated flattened carry and y list,
  # we need to know how many elements out of that list is the carry.
  flat_carry_len = len(flat_init)
  flat_xs_len = len(flat_xs)

  # `fn` operates on PyTrees and returns PyTrees. However, XLA only understands
  # (lists or tuples of) tensors. So we will craft a `flat_fn` that takes in
  # flattened PyTrees, internally recreates the desired tree structure, then
  # calls `fn`. Similar transformation on the return path.
  def flat_fn(*seq: torch.Tensor) -> Sequence[torch.Tensor]:
    carry = seq[:flat_carry_len]
    x = seq[flat_carry_len:]
    carry_pytree = tree_unflatten(carry, carry_spec)
    x_pytree = tree_unflatten(x, xs_spec)
    carry_pytree, y_pytree = fn(carry_pytree, x_pytree)
    carry, _ = tree_flatten(carry_pytree)
    y, _ = tree_flatten(y_pytree)
    return carry + y

  # Abstractly trace and lower `fn`.
  # Later we will include `fn_computation` within the while loop body.
  def make_fake_tensor(v: torch.Tensor) -> torch.Tensor:
    # TODO: there are some problems in PyTorch/XLA, or I'm missing something, because
    #
    #    torch.empty(v.size(), dtype=v.dtype, requires_grad=v.requires_grad).to(device)
    #
    # results in a tensor without gradient tracking, and
    #
    #    torch.empty(v.size(), dtype=v.dtype, device=v.device, requires_grad=v.requires_grad)
    #
    # results in incorrect calculation, unless they are `print()`-ed.
    # But all three should be equivalent.
    t = torch.empty(v.size(), dtype=v.dtype).to(device)
    t.requires_grad_(v.requires_grad)
    return t

  device = torch_xla.device()
  fake_carry_pytree = tree_map(make_fake_tensor, init)
  fake_x_pytree = tree_map(lambda v: make_fake_tensor(v[0]), xs)
  fn_output_carry_pytree, fn_output_y_pytree = fn(fake_carry_pytree,
                                                  fake_x_pytree)
  # Later we'll use `fn_output_carry_spec` etc to turn flattened outputs back to a PyTree.
  _, fn_output_carry_spec = tree_flatten(fn_output_carry_pytree)
  assert fn_output_carry_spec == carry_spec
  fake_output_y, fn_output_y_spec = tree_flatten(fn_output_y_pytree)
  assert len(fake_output_y) == flat_carry_len
  fake_carry, _ = tree_flatten(fake_carry_pytree)
  fake_x, _ = tree_flatten(fake_x_pytree)
  fn_outputs = flat_fn(*(fake_carry + fake_x))
  fn_ctx = torch_xla._XLAC.lowering.LoweringContext()
  fn_ctx.set_name_string("my_ctx")
  fn_ctx.build(list(fn_outputs))
  fn_hlo = fn_ctx.hlo()
  fn_computation = xb.computation_from_module_proto("my_fn_computation", fn_hlo)

  builder = Builder('scan')

  # Figure out the shape of `ys` from the abstract tracing.
  fn_carry_out = fn_outputs[:flat_carry_len]
  fn_y_out = fn_outputs[flat_carry_len:]
  fn_carry_shapes = [v.shape for v in fn_carry_out]
  fn_y_shapes = [v.shape for v in fn_y_out]
  for fn_carry_shape, init_leaf in zip(fn_carry_shapes, flat_init):
    assert fn_carry_shape == init_leaf.shape, f"`fn` must keep the `carry` shape unchanged. \
      Got {fn_carry_shape} but expected {init_leaf.shape}"

  # Since we are threading four PyTrees through the body_fn:
  # - carry: the scan state
  # - xs: the flattened input pytree
  # - fn_carry_history: history of that state
  # - ys: the flattened output of fn
  #
  # We need to concatenate all three into one big list prior to
  # entering `body_fn` and `cond_fn`, and split them back to three
  # objects which is easier to work with after that. This pair of
  # functions is for that purpose.
  T = TypeVar('T')

  def pack(carry: Sequence[T], xs: Sequence[T], fn_carry_history: Sequence[T],
           ys: Sequence[T]) -> Tuple[T, ...]:
    return tuple(carry) + tuple(xs) + tuple(fn_carry_history) + tuple(ys)

  def unpack(seq: Sequence[T]) -> Tuple[List[T], List[T], List[T], List[T]]:
    seq = list(seq)
    carry = seq[:flat_carry_len]
    xs = seq[flat_carry_len:flat_carry_len + flat_xs_len]
    fn_carry_history = seq[flat_carry_len + flat_xs_len:flat_carry_len * 2 +
                           flat_xs_len]
    ys = seq[flat_carry_len * 2 + flat_xs_len:]
    return carry, xs, fn_carry_history, ys

  xs_len = fake_x[0].shape[0]
  num_iters = torch.tensor(xs_len, device=device)
  ys = [
      torch.zeros((xs_len, *fn_y_shape), device=device)
      for fn_y_shape in fn_y_shapes
  ]
  fn_carry_history = [
      torch.zeros((xs_len, *fn_carry_shape), device=device)
      for fn_carry_shape in fn_carry_shapes
  ]
  loop_tensors: Tuple[torch.Tensor, ...] = (num_iters,) + pack(
      flat_init, flat_xs, fn_carry_history, ys)
  for val in loop_tensors:
    builder.add_param(val)

  # If there are additional device data tensors referenced by the computation that
  # are not input or carry, we need to provide those tensors when calling
  # `fn_computation`. As a result, we need to determine what are those tensors and they
  # need to be provided as additional inputs to `cond_fn` and `body_fn`.

  # Add additional inputs as params as well.
  mapping: Dict[int, torch.Tensor] = fn_ctx.parameter_id_tensor_mapping()
  param_id_to_additional_tensors_param_id: Dict[int, int] = {}
  num_params = len(mapping)
  for v in itertools.chain(fake_carry, fake_x):
    param_id = fn_ctx.tensor_parameter_id(v)
    if param_id != -1:
      del mapping[param_id]
  for param_id in range(num_params):
    if param_id in mapping:
      idx = builder.add_param(mapping[param_id].to(torch_xla.device()))
      param_id_to_additional_tensors_param_id[param_id] = idx
  num_additional_inputs = len(mapping)

  def skip_additional_inputs(fn):

    def wrapper(*args):
      first_args = args[:builder.num_params() - num_additional_inputs]
      return fn(*first_args)

    return wrapper

  def pass_through_additional_inputs(fn):

    def wrapper(*args):
      first_args = args[:builder.num_params() - num_additional_inputs]
      additional_inputs = args[builder.num_params() - num_additional_inputs:]
      res = fn(*first_args, additional_inputs=additional_inputs)
      assert isinstance(res, tuple)
      return xb.Op.tuple(res + additional_inputs)

    return wrapper

  # For each tensor, we need to know its parameter ID.
  # Then we should order the tensors in increasing parameter ID order when passing them
  # to `xb.Op.call`. For each (ID, tensor) in `mapping`:
  # - Check if the tensor is a fake tensor we just created
  # - If yes, find the position in the fake tensor list. Index into the input ops.
  # - If no, find the op in the additional input list.
  def call_fn_computation(carry: List[xb.Op], x: List[xb.Op],
                          additional_inputs: Tuple[xb.Op, ...]) -> xb.Op:
    param_id_to_fake_tensors_id: Dict[int, int] = {}
    for i, v in enumerate(itertools.chain(fake_carry, fake_x)):
      param_id = fn_ctx.tensor_parameter_id(v)
      if param_id != -1:
        param_id_to_fake_tensors_id[param_id] = i

    all_inputs = carry + x
    all_inputs_reordered = []
    mapping: Dict[int, torch.Tensor] = fn_ctx.parameter_id_tensor_mapping()
    for i in range(len(mapping)):
      if i in param_id_to_fake_tensors_id:
        op = all_inputs[param_id_to_fake_tensors_id[i]]
        all_inputs_reordered.append(op)
      else:
        # TODO: super subtle. what is the right abstraction for this?
        op = additional_inputs[param_id_to_additional_tensors_param_id[i] -
                               len(loop_tensors)]
        all_inputs_reordered.append(op)
    return xb.Op.call(fn_computation, all_inputs_reordered)

  @skip_additional_inputs
  def cond_fn(num_iters: xb.Op, *args: xb.Op):
    return num_iters > xb.Op.scalar(num_iters.builder(), 0, dtype=xb.Type.S64)

  @pass_through_additional_inputs
  def body_fn(num_iters: xb.Op, *args: xb.Op, additional_inputs: Tuple[xb.Op,
                                                                       ...]):
    carry, xs, fn_carry_history, ys = unpack(args)
    xs_len_op = xb.Op.scalar(num_iters.builder(), xs_len, dtype=xb.Type.S64)
    one = xb.Op.scalar(num_iters.builder(), 1, dtype=xb.Type.S64)
    idx = xs_len_op - num_iters
    x = [dynamic_slice(v, idx) for v in xs]
    for i in range(len(carry)):
      fn_carry_history[i] = dynamic_update_slice(fn_carry_history[i], carry[i],
                                                 idx)
    result = call_fn_computation(carry, x, additional_inputs)
    for i in range(flat_carry_len):
      carry[i] = result.get_tuple_element(i)
      y = result.get_tuple_element(i + flat_carry_len)
      ys[i] = dynamic_update_slice(ys[i], y, idx)
    return (num_iters - one,) + pack(carry, xs, fn_carry_history, ys)

  res = xb.Op.mkwhile(builder.params(), cond_fn, body_fn)
  computation = res.build('scan')

  outputs = torch_xla._XLAC._xla_user_computation('xla::scan',
                                                  builder.param_tensors(),
                                                  computation)
  # skip the last num_additional_inputs
  outputs = outputs[:len(outputs) - num_additional_inputs]
  # `1:` to skip `num_iters`
  carry, xs, fn_carry_history, ys = unpack(outputs[1:])

  # Unflatten tensors back to PyTrees
  return tree_unflatten(carry, carry_spec), tree_unflatten(
      fn_carry_history, carry_spec), tree_unflatten(ys, fn_output_y_spec)


import torch.autograd
from torch.utils.checkpoint import detach_variable


class Scan(torch.autograd.Function):

  @staticmethod
  def forward(ctx, fn, init, xs):
    # Forward pass, save inputs for backward
    ctx._fn = fn
    with torch._C._AutoDispatchBelowAutograd():
      carry, carry_history, ys = _scan_carry_history(fn, init, xs)
    flat_carry_history, carry_spec = tree_flatten(carry_history)
    flat_xs, xs_spec = tree_flatten(xs)
    ctx.save_for_backward(*flat_carry_history, *flat_xs)
    ctx._flat_carry_len = len(flat_carry_history)
    ctx._carry_spec = carry_spec
    ctx._xs_spec = xs_spec
    return carry, carry_history, ys

  @staticmethod
  def backward(ctx, grad_carry, grad_carry_history, grad_ys):
    fn = ctx._fn
    flat_carry_len = ctx._flat_carry_len
    carry_spec = ctx._carry_spec
    xs_spec = ctx._xs_spec
    tensors_list = ctx.saved_tensors
    carry_history = tree_unflatten(tensors_list[:flat_carry_len], carry_spec)
    xs = tree_unflatten(tensors_list[flat_carry_len:], xs_spec)

    def step_fn(grad_carry, pytree: Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]):
      grad_y, carry, x = pytree
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
    carry_history = carry_history.flip(0).requires_grad_(True)
    xs = xs.flip(0).requires_grad_(True)
    grad_ys = grad_ys.flip(0).requires_grad_(True)

    grad_init, _, grad_xs = scan_dense(step_fn, grad_init,
                                       (grad_ys, carry_history, xs))

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
  import numpy as np
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
