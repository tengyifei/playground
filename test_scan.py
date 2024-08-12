import torch_xla
import torch
from torch.utils._pytree import tree_map
import numpy as np
import pytest

from scan_prototype import scan

device = torch_xla.device()


def test_scan_forward():

  # A simple function to be applied at each step of the scan
  def step_fn(carry, x):
    new_carry = carry + x
    y = carry * x
    return new_carry, y

  # Initial carry
  init_carry = torch.tensor([1.0, 1.0, 1.0], requires_grad=False, device=device)

  # Example input tensor of shape (batch_size, features)
  xs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    requires_grad=True,
                    device=device)

  # Use the scan function
  final_carry, ys = scan(step_fn, init_carry, xs)

  # Loss for backward pass (sum of the outputs)
  loss = ys.sum()
  torch_xla.sync()
  print(loss)
  assert loss.item() == 249.0


def test_scan_autograd():

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


# Hypothetical `scan` function that supports PyTrees
def scan_brute_force_pytree(fn, init, xs):
  carry = init
  ys = []

  for i in range(len(xs[0])):
    carry, y = fn(carry, tree_map(lambda x: x[i], xs))
    ys.append(y)

  # Stack the results of y (if it's a tensor) into a single tensor
  ys = tree_map(lambda *x: torch.stack(x), *ys)
  return carry, ys


@pytest.mark.parametrize("scan_fn", [scan_brute_force_pytree, scan])
def test_scan_pytree_forward(scan_fn):
  # Step function that operates on a tuple (carry, (x1, x2)) where x1 and x2 have different sizes
  def step_fn(carry, x):
    carry1, carry2 = carry
    x1, x2 = x

    new_carry1 = carry1 + x1.sum()
    new_carry2 = carry2 + x2.sum()

    y1 = x1 * 2
    y2 = x2 * 2

    return (new_carry1, new_carry2), (y1, y2)

  # Initial carry: tuple of tensors with different sizes
  init_carry = (torch.tensor([0.0], device=device),
                torch.tensor([1.0, 2.0], device=device))

  # Example input: tuple of tensors with different sizes
  xs = (
      torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),  # Shape (2, 2)
      torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
                   device=device)  # Shape (2, 3)
  )

  # Call the scan function
  final_carry, ys = scan_fn(step_fn, init_carry, xs)

  # Print the outputs
  torch_xla.sync()
  print("Final carry:", final_carry)
  print("Outputs ys:", ys)

  # Expected values from the PyTorch script output
  expected_final_carry = (np.array([10.0]), np.array([46.0, 47.0]))
  expected_ys_1 = np.array([[2.0, 4.0], [6.0, 8.0]])
  expected_ys_2 = np.array([[10.0, 12.0, 14.0], [16.0, 18.0, 20.0]])

  # Convert PyTorch tensors to numpy arrays for comparison
  final_carry_np = (final_carry[0].cpu().numpy(), final_carry[1].cpu().numpy())
  ys_1_np, ys_2_np = ys[0].cpu().numpy(), ys[1].cpu().numpy()

  # Assert statements to verify the output
  assert np.allclose(
      final_carry_np[0], expected_final_carry[0]
  ), f"Final carry[0] mismatch: {final_carry_np[0]} != {expected_final_carry[0]}"
  assert np.allclose(
      final_carry_np[1], expected_final_carry[1]
  ), f"Final carry[1] mismatch: {final_carry_np[1]} != {expected_final_carry[1]}"

  assert np.allclose(
      ys_1_np,
      expected_ys_1), f"Outputs ys[0] mismatch: {ys_1_np} != {expected_ys_1}"
  assert np.allclose(
      ys_2_np,
      expected_ys_2), f"Outputs ys[1] mismatch: {ys_2_np} != {expected_ys_2}"

  print("All assertions passed!")
