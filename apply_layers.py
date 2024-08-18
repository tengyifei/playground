import torch_xla
from typing import Iterable
import torch.nn as nn
import torch
from torch.utils._pytree import tree_map

from scan_prototype import scan


def extract_weights_dict(module: nn.Module):
  """
  Extracts the parameters (weights and biases) from a PyTorch module and stores them in a dictionary.
  """
  weights_dict = {
      name: param.clone() for name, param in module.named_parameters()
  }
  return weights_dict


def apply_weights_dict(module: nn.Module, weights_dict):
  """
  Re-applies the weights and biases from the dictionary back to the PyTorch module.
  """
  for name, param in module.named_parameters():
    if name in weights_dict:
      torch.utils.swap_tensors(param, weights_dict[name].clone())


def apply_layers(layers: Iterable[torch.nn.Module], input_data):
  # Extract and stack the parameters into a pytree
  params = [extract_weights_dict(layer) for layer in layers]
  stacked_params = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                            *params)

  # Empty layers case.
  if not params:
    return input_data

  # Use the first layer as the example/template layer
  from copy import deepcopy
  example_layer = deepcopy(next(iter(layers)))

  # Hollow out the weights and biases in the example layer
  example_layer = example_layer.to_empty(device=None)

  # Function to apply at each step
  def one_layer(carry, params):
    # Apply the current layer's weights and biases to the example layer and run
    apply_weights_dict(example_layer, params)
    return example_layer(carry), example_layer(carry) * 0

  final_carry, _ = scan(one_layer, input_data, stacked_params)

  return final_carry


def loopy_apply_layers(layers: Iterable[torch.nn.Module], input_data):
  # Extract and stack the parameters into a pytree
  params = [extract_weights_dict(layer) for layer in layers]
  stacked_params = tree_map(lambda *tensors: torch.stack(tensors, dim=0),
                            *params)

  # Empty layers case.
  if not params:
    return input_data

  # Use the first layer as the example/template layer
  from copy import deepcopy
  example_layer = deepcopy(next(iter(layers)))

  # Hollow out the weights and biases in the example layer
  example_layer = example_layer.to_empty(device=None)

  # Function to apply at each step
  def one_layer(carry, params):
    # Apply the current layer's weights and biases to the example layer and run
    apply_weights_dict(example_layer, params)
    return example_layer(carry), example_layer(carry) * 0

  final_carry, _ = _loopy_scan(one_layer, input_data, stacked_params)

  return final_carry


def _loopy_scan(fn, init, xs):
  """A simple scan implemented with for loops serving as reference
  implementation."""
  from torch.utils._pytree import tree_map, tree_iter
  carry = init
  ys = []
  xs_len = len(next(iter(tree_iter(xs))))
  for i in range(xs_len):
    carry, y = fn(carry, tree_map(lambda x: x[i], xs))
    ys.append(y)
  ys = tree_map(lambda *x: torch.stack(x), *ys)
  return carry, ys
