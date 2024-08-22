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


def test_scan_linear_layers():
  import torch_xla
  import torch.nn as nn
  from apply_layers import apply_layers

  device = torch_xla.device()

  # We want to apply these layers sequentially
  import torch.nn as nn
  layers = [nn.Linear(64, 64).to(device) for _ in range(10)]
  input_data = torch.randn(64).to(device)

  from copy import deepcopy
  scan_layers = deepcopy(layers)
  loop_layers = deepcopy(layers)

  torch_xla.sync()

  output = apply_layers(scan_layers, input_data.clone())
  print("Output:", output)
  output.sum().backward()

  # Test that the result is the same as for loop.
  loop_output = input_data.clone()
  from copy import deepcopy
  for layer in loop_layers:
    loop_output = layer(loop_output)
  print("Loop output:", loop_output)
  import numpy as np
  assert np.allclose(
      loop_output.detach().cpu().numpy(),
      output.detach().cpu().numpy(),
      atol=0.0001,
      rtol=0.01)

  loop_output.sum().backward()

  # Test that the gradients are the same too.
  for layer_scan, layer_loop in zip(scan_layers, loop_layers):
    assert np.allclose(
        layer_scan.weight.grad.detach().cpu().numpy(),  # type: ignore
        layer_loop.weight.grad.detach().cpu().numpy(),  # type: ignore
        atol=0.0001,
        rtol=0.01), f"{layer_scan.weight.grad} != {layer_loop.weight.grad}"
    assert np.allclose(
        layer_scan.bias.grad.detach().cpu().numpy(),  # type: ignore
        layer_loop.bias.grad.detach().cpu().numpy(),  # type: ignore
        atol=0.0001,
        rtol=0.01), f"{layer_scan.bias.grad} != {layer_loop.bias.grad}"


def test_scan_decoder_model():
  import torch_xla
  from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

  device = torch_xla.device()

  with torch.no_grad():
    # Define the configuration
    config = DecoderOnlyConfig()

    # Instantiate the model
    model = DecoderOnlyModel(config).to(device)

    # Set the batch size and sequence length
    batch_size = 2  # 2 sequences in parallel
    sequence_length = 10  # each sequence is 10 tokens long

    # Generate random input_ids within the range of the vocabulary size
    torch.random.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size,
                              (batch_size, sequence_length)).to(device)

    # Feed the input_ids into the model
    loop_output = model(input_ids.clone())
    print(f"Loop output shape: {loop_output.shape}")

    # Run again, this time using `scan`
    scan_output = model.forward_scan(input_ids.clone())
    print(f"Scan output shape: {scan_output.shape}")

    torch_xla.sync(wait=True)
    close = np.allclose(
        loop_output.detach().cpu().numpy(),
        scan_output.detach().cpu().numpy(),
        atol=0.05,
        rtol=0.01)
    assert close


def test_scan_decoder_model_autograd():
  import torch_xla
  from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

  device = torch_xla.device()

  # Define the configuration
  config = DecoderOnlyConfig()

  # Instantiate the model
  model = DecoderOnlyModel(config).to(device)

  # Set the batch size and sequence length
  batch_size = 2  # 2 sequences in parallel
  sequence_length = 10  # each sequence is 10 tokens long

  # Generate random input_ids within the range of the vocabulary size
  torch.random.manual_seed(1)
  input_ids = torch.randint(0, config.vocab_size,
                            (batch_size, sequence_length)).to(device)

  from copy import deepcopy
  loop_model = deepcopy(model)
  scan_model = deepcopy(model)

  # Feed the input_ids into the model
  loop_output = loop_model(input_ids.clone())
  loop_output.sum().backward()
  torch_xla.sync(wait=True)

  # Run again, this time using `scan`
  scan_output = scan_model.forward_scan(input_ids.clone())

  scan_output.sum().backward()
  torch_xla.sync(wait=True)

  close = np.allclose(
      loop_output.detach().cpu().numpy(),
      scan_output.detach().cpu().numpy(),
      atol=0.05,
      rtol=0.01)
  assert close

  # Check gradients
  for layer_scan, layer_loop in zip(scan_model.layers, loop_model.layers):
    for (name, param_scan), (name2,
                             param_loop) in zip(layer_scan.named_parameters(),
                                                layer_loop.named_parameters()):
      assert name == name2
      if param_scan.grad is not None or param_loop.grad is not None:
        scan_grad = param_scan.grad.detach().cpu().numpy()  # type: ignore
        loop_grad = param_loop.grad.detach().cpu().numpy()  # type: ignore
        assert np.allclose(
            scan_grad, loop_grad, atol=0.1,
            rtol=0.05), f"{name} gradient mismatch: {scan_grad} != {loop_grad}"
        print(f"Pass: {name} {param_scan.shape}")
