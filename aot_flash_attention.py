import torch
import torch_xla
from typing import List
import functools
import os
import warnings

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.debug.metrics as met

from typing import Any, List, Callable, Optional, Tuple, Dict
from torch.library import impl, custom_op
from torch_xla.core.xla_model import XLA_LIB
from torch_xla.experimental.custom_kernel import FlashAttention
import torch_xla.debug.profiler as xp

_XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0") == "1"

_DEBUG = False


def describe_value(v):
  if v is not None and isinstance(v, torch.Tensor):
    print(f"{type(v)}({v.shape}, dtype={v.dtype}, device={v.device})")
  elif isinstance(v, list):
    print(f"list({len(v)})")
  elif v is None:
    print("None")
  else:
    print(type(v))


def _extract_backend_config(
    module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> Optional[str]:
  """
  This algorithm intends to extract the backend config from the compiler IR like the following,
  and it is not designed to traverse any generic MLIR module.

  module @jit_add_vectors attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
    func.func public @main(%arg0: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<8xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @add_vectors(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @add_vectors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @wrapped(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @wrapped(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @apply_kernel(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @apply_kernel(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSMTkuMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA3lZDQFVBwsPEw8PCw8PMwsLCwtlCwsLCwsPCw8PFw8LFw8PCxcPCxcTCw8LDxcLBQNhBwNZAQ0bBxMPGw8CagMfBRcdKy0DAycpHVMREQsBBRkVMzkVTw8DCxUXGRsfCyELIyUFGwEBBR0NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFHwUhBSMFJQUnEQMBBSkVLw8dDTEXA8IfAR01NwUrFwPWHwEVO0EdPT8FLRcD9h8BHUNFBS8XA3InAQMDSVcFMR1NEQUzHQ1RFwPGHwEFNSN0cHUubWVtb3J5X3NwYWNlPHZtZW0+ACNhcml0aC5vdmVyZmxvdzxub25lPgAXVQMhBx0DJwMhBwECAgUHAQEBAQECBASpBQEQAQcDAQUDEQETBwMVJwcBAQEBAQEHAwUHAwMLBgUDBQUBBwcDBQcDAwsGBQMFBQMLCQdLRwMFBQkNBwMJBwMDCwYJAwUFBRENBAkHDwURBQABBgMBBQEAxgg32wsdE2EZ2Q0LEyMhHSknaw0LCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAc3RvcmUAL3dvcmtzcGFjZXMvd29yay9weXRvcmNoL3hsYS90ZXN0L3Rlc3Rfb3BlcmF0aW9ucy5weQBhZGRfdmVjdG9yc19rZXJuZWwAZGltZW5zaW9uX3NlbWFudGljcwBmdW5jdGlvbl90eXBlAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAHN5bV9uYW1lAG1haW4AdmFsdWUAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgOCldLCBbXSksKSksICg4LCksICgpKV0sIFtdKSwpKV0AYWRkX3ZlY3RvcnMAdGVzdF90cHVfY3VzdG9tX2NhbGxfcGFsbGFzX2V4dHJhY3RfYWRkX3BheWxvYWQAPG1vZHVsZT4Ab3ZlcmZsb3dGbGFncwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\22, \22needs_layout_passes\22: true}}", kernel_name = "add_vectors_kernel", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
  }

  Basically, what we are looking for is a two level of operations, and the tpu_custom_call operation in the inner level. It will return None if the payload is not found.
  """
  for operation in module.body.operations:
    assert len(
        operation.body.blocks) == 1, "The passing module is not compatible."
    for op in operation.body.blocks[0].operations:
      if op.name == "stablehlo.custom_call":
        return op.backend_config.value
  return None


def jax_import_guard():
  # Somehow, we need to grab the TPU before JAX locks it. Otherwise, any pt-xla TPU operations will hang.
  torch_xla._XLAC._init_computation_client()


def convert_torch_dtype_to_jax(dtype: torch.dtype) -> "jnp.dtype":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax.numpy as jnp
  if _XLA_USE_BF16:
    raise RuntimeError(
        "Pallas kernel does not support XLA_USE_BF16, please unset the env var")
  if dtype == torch.float32:
    return jnp.float32
  elif dtype == torch.float64:
    return jnp.float64
  elif dtype == torch.float16:
    return jnp.float16
  elif dtype == torch.bfloat16:
    return jnp.bfloat16
  elif dtype == torch.int32:
    return jnp.int32
  elif dtype == torch.int64:
    return jnp.int64
  elif dtype == torch.int16:
    return jnp.int16
  elif dtype == torch.int8:
    return jnp.int8
  elif dtype == torch.uint8:
    return jnp.uint8
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")


def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax

  return jax.ShapeDtypeStruct(tensor.shape,
                              convert_torch_dtype_to_jax(tensor.dtype))


trace_pallas_arg_to_payload: Dict[Tuple[Any], str] = {}


def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 use_cache=False,
                 **kwargs):
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax._src.pallas.mosaic.pallas_call_registration

  jax_args = []  # for tracing
  tensor_args = []  # for execution
  for i, arg in enumerate(args):
    # TODO: Could the args be a tuple of tensors or a list of tensors? Flattern them?
    if torch.is_tensor(arg):
      # ShapeDtypeStruct doesn't have any storage and thus is very suitable for generating the payload.
      jax_meta_tensor = to_jax_shape_dtype_struct(arg)
      jax_args.append(jax_meta_tensor)
      tensor_args.append(arg)
    else:
      jax_args.append(arg)

  hash_key = ()
  if use_cache:
    global trace_pallas_arg_to_payload
    # implcit assumption here that everything in kwargs is hashable and not a tensor,
    # which is true for the gmm and tgmm.
    hash_key = (jax.config.jax_default_matmul_precision, kernel, static_argnums,
                tuple(static_argnames)
                if static_argnames is not None else static_argnames,
                tuple(jax_args), repr(sorted(kwargs.items())).encode())
    if hash_key in trace_pallas_arg_to_payload:
      torch_xla._XLAC._xla_increment_counter('trace_pallas_cache_hit', 1)
      return trace_pallas_arg_to_payload[hash_key], tensor_args

  # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.
  ir = jax.jit(
      kernel, static_argnums=static_argnums,
      static_argnames=static_argnames).lower(*jax_args, **kwargs).compiler_ir()
  payload = _extract_backend_config(ir)

  if use_cache:
    # if we reach here it means we have a cache miss.
    trace_pallas_arg_to_payload[hash_key] = payload

  return payload, tensor_args


def make_kernel_from_pallas(kernel: Callable, output_shape_dtype_fn: Callable):
  # TODO: Maybe we can cache the payload for the same input.
  def wrapped_kernel(kernel: Callable,
                     output_shape_dtype_fn: Callable,
                     *args,
                     static_argnums=None,
                     static_argnames=None,
                     **kwargs) -> Callable:
    payload, tensor_args = trace_pallas(
        kernel,
        *args,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        **kwargs)
    output_shape_dtype = output_shape_dtype_fn(*args)
    assert isinstance(output_shape_dtype,
                      list), "The output_shape_dtype_fn should return a list."
    output_shapes = [shape for shape, _ in output_shape_dtype]
    output_dtypes = [dtype for _, dtype in output_shape_dtype]
    outputs = torch_xla._XLAC._xla_tpu_custom_call(tensor_args, payload,
                                                   output_shapes, output_dtypes)

    # Make the output easier to use.
    if len(outputs) == 1:
      return outputs[0]
    return tuple(outputs)

  return functools.partial(wrapped_kernel, kernel, output_shape_dtype_fn)


# Note: the alias inference and mutation removal in PyTorch doesn't work. So we
#
# - Explicitly clone all inputs.
# - Clone outputs if the output aliases an input.
#
@custom_op("xla::fa_custom_forward", mutates_args=())
def fa_custom_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
  partition_spec = ('fsdp', 'tensor', None, None)
  mesh = xs.get_global_mesh()
  assert mesh is not None

  if _DEBUG:
    print("Inside fa_custom_forward")
    for t in [q, k, v]:
      describe_value(t)

  q_segment_ids = kv_segment_ids = ab = None
  sm_scale = 1.0
  causal = True

  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl

  q_full_shape = None
  kv_full_shape = None
  save_residuals = True

  # SPMD integration.
  # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
  # PyTorch tell us clone is necessary:
  #
  # RuntimeError: Found a custom (non-ATen) operator whose output has alias
  # annotations: xla::fa_custom_forward(Tensor(a0!) q, Tensor(a1!) k,
  # Tensor(a2!) v) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor). We only
  # support functionalizing operators whose outputs do not have alias
  # annotations (e.g. 'Tensor(a)' is a Tensor with an alias annotation whereas
  # 'Tensor' is a Tensor without. The '(a)' is the alias annotation). The alias
  # annotation specifies that the output Tensor shares storage with an input
  # that has the same annotation. Please check if (1) the output needs to be an
  # output (if not, don't return it), (2) if the output doesn't share storage
  # with any inputs, then delete the alias annotation. (3) if the output indeed
  # shares storage with an input, then add a .clone() before returning it to
  # prevent storage sharing and then delete the alias annotation. Otherwise,
  # please file an issue on GitHub.
  #
  with xp.Trace('shard'):
    full_q = q.clone()
    full_k = k.clone()
    full_v = v.clone()
    full_ab = ab
    if partition_spec is not None:
      q_full_shape = q.shape
      kv_full_shape = k.shape
      q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
      k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
      v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
      if ab:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor

  # It computes the shape and type of o, l, m.
  shapes = [q.shape]
  dtypes = [q.dtype]
  if save_residuals:
    res_shape = list(q.shape)
    res_shape[-1] = FlashAttention.MIN_BLOCK_SIZE
    for _ in range(2):
      shapes.append(res_shape)
      dtypes.append(torch.float32)

  with torch.no_grad():
    if partition_spec is not None and q_segment_ids is not None and kv_segment_ids is not None:
      # partition_spec is for q,k,v with shape [batch, num_head, seq_len, head_dim], segment id
      # is of shape [batch, seq_len], hence we need to tweak it a bit
      segment_id_partition_spec = (partition_spec[0], partition_spec[2])
      q_segment_ids = xs.enable_manual_sharding(
          q_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
      kv_segment_ids = xs.enable_manual_sharding(
          kv_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
    segment_ids, q_segment_ids_fa, kv_segment_ids_fa = FlashAttention.prepare_segment_ids(
        q_segment_ids, kv_segment_ids)

    with xp.Trace('pallas'):
      # We can't directly use flash_attention as we need to override the save_residuals flag which returns
      # l and m that is needed for the backward. Then we lose all the shape checks.
      # TODO: replicate the shape checks on flash_attention.
      # Here we seperate the tracing and execution part just to support SegmentIds.
      payload, _ = trace_pallas(
          _flash_attention_impl,
          q,
          k,
          v,
          ab,
          segment_ids,
          save_residuals,
          causal,
          sm_scale,
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
          min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
          False,
          static_argnums=range(5, 13),
          use_cache=True,
      )

    with xp.Trace('custom_call'):
      args = [q, k, v]
      if ab is not None:
        args += [ab]
      if segment_ids is not None:
        args += [q_segment_ids_fa, kv_segment_ids_fa]
      o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)

    if not save_residuals:
      o = o[0]
      # SPMD integration
      if partition_spec is not None:
        o = xs.disable_manual_sharding(
            o, partition_spec, q_full_shape, mesh=mesh).global_tensor
      return o

    assert isinstance(o, list)
    o, *aux = o

    # The fancier slice notation lowers to `aten.take`, which sends a large indexing
    # tensor to the device and confuses the XLA compiler when used under scan for some reason.
    # See the transfer to device in a trace: http://shortn/_4zOQhGezCS.
    # As a result, we get a `!IsManual()` assertion in HLO sharding propgation.
    # Therefore, we spell it as a permute + index into the first dim.
    # However, that causes NaN loss for some reason. So we'll perform the slicing instead.
    # l = aux[-2][:, :, :, 0]
    # l = aux[-2].permute(3, 0, 1, 2)[0]
    l = aux[-2]
    l = torch.ops.aten.slice(l, -1, 0, 1)
    # print(torch_xla._XLAC._get_xla_tensors_text([l]))
    # m = aux[-1][:, :, :, 0]
    # m = aux[-1].permute(3, 0, 1, 2)[0]
    m = aux[-1]
    m = torch.ops.aten.slice(m, -1, 0, 1)

  # SPMD integration
  with xp.Trace('index_lm'):
    if partition_spec is not None:
      o = xs.disable_manual_sharding(
          o, partition_spec, q_full_shape, mesh=mesh).global_tensor
      l = xs.disable_manual_sharding(
          l, partition_spec, q_full_shape[:3] + (l.shape[-1],),
          mesh=mesh).global_tensor
      m = xs.disable_manual_sharding(
          m, partition_spec, q_full_shape[:3] + (m.shape[-1],),
          mesh=mesh).global_tensor

    l = l.squeeze(-1)
    m = m.squeeze(-1)

  assert partition_spec is not None

  # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
  # but it should be OK as the backward will use the same partition_spec
  outs = [o] + [full_q, full_k, full_v, l, m]
  if _DEBUG:
    print("Outs")
    for t in outs:
      describe_value(t)
  return tuple(outs)


@fa_custom_forward.register_fake
def fa_custom_forward_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
  if _DEBUG:
    print("Inside fake fa_custom_forward")

  assert q.shape == k.shape
  assert k.shape == v.shape

  # full_q, full_k, full_v, o, l, m
  full_q = torch.empty_like(q)
  full_k = torch.empty_like(k)
  full_v = torch.empty_like(v)
  o = torch.empty_like(v)
  l = torch.empty_like(v, dtype=torch.float32)[..., 0]
  m = torch.empty_like(v, dtype=torch.float32)[..., 0]

  return tuple([torch.empty_like(o)] +
               [torch.empty_like(t) for t in (
                   full_q,
                   full_k,
                   full_v,
                   l,
                   m,
               )])


def defeat_alias(v):
  return v * 1.0


@custom_op("xla::fa_custom_backward", mutates_args=())
def fa_custom_backward(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, o: torch.Tensor, l: torch.Tensor, m: torch.Tensor,
    q_shape: List[int],
    k_shape: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  q_segment_ids_fa = kv_segment_ids_fa = ab = None

  partition_spec = ('fsdp', 'tensor', None, None)
  mesh = xs.get_global_mesh()
  assert mesh is not None

  if _DEBUG:
    print("Inside fa_custom_backward")

  from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv

  grad_output = defeat_alias(grad_output)
  saved_tensors = (q, k, v, o, l, m)
  q, k, v, o, l, m = (defeat_alias(t) for t in saved_tensors)

  causal = True
  sm_scale = 1.0
  q_full_shape = torch.Size(q_shape)
  kv_full_shape = torch.Size(k_shape)
  # this segment_ids only reflects the local shape of segment_ids
  segment_ids = None
  grad_q = grad_k = grad_v = grad_ab = None
  needs_input_grad = [True, True, True]
  grad_i = torch.sum(
      o.to(torch.float32) * grad_output.to(torch.float32),
      axis=-1)  # [batch_size, num_heads, q_seq_len]

  expanded_l = l.unsqueeze(-1).expand([-1 for _ in l.shape] +
                                      [FlashAttention.MIN_BLOCK_SIZE])
  expanded_m = m.unsqueeze(-1).expand([-1 for _ in m.shape] +
                                      [FlashAttention.MIN_BLOCK_SIZE])
  expanded_grad_i = grad_i.unsqueeze(-1).expand([-1 for _ in grad_i.shape] +
                                                [FlashAttention.MIN_BLOCK_SIZE])

  # SPMD integration
  if partition_spec is not None:
    q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
    k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
    v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
    expanded_l = xs.enable_manual_sharding(
        expanded_l, partition_spec, mesh=mesh).global_tensor
    expanded_m = xs.enable_manual_sharding(
        expanded_m, partition_spec, mesh=mesh).global_tensor
    grad_output = xs.enable_manual_sharding(
        grad_output, partition_spec, mesh=mesh).global_tensor
    expanded_grad_i = xs.enable_manual_sharding(
        expanded_grad_i, partition_spec, mesh=mesh).global_tensor
    if ab:
      ab = xs.enable_manual_sharding(
          ab, partition_spec, mesh=mesh).global_tensor

  if needs_input_grad[0]:
    payload, _ = trace_pallas(
        _flash_attention_bwd_dq,
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        grad_output,
        grad_i,
        block_q_major=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dq"],
                          q.shape[2]),
        block_k_major=min(
            FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dq"], k.shape[2]),
        block_k=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dq"],
                    k.shape[2]),
        sm_scale=sm_scale,
        causal=causal,
        mask_value=FlashAttention.DEFAULT_MASK_VALUE,
        debug=False,
        static_argnames=[
            "block_q_major", "block_k_major", "block_k", "sm_scale", "causal",
            "mask_value", "debug"
        ],
        use_cache=True,
    )

    args = [q, k, v]
    if ab is not None:
      args += [ab]
    if segment_ids is not None:
      args += [q_segment_ids_fa, kv_segment_ids_fa]
    args += [expanded_l, expanded_m, grad_output, expanded_grad_i]

    outputs = [q]
    if ab is not None:
      outputs += [ab]
    grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                 [i.shape for i in outputs],
                                                 [i.dtype for i in outputs])
    if needs_input_grad[0]:
      grad_q = grads[0]

  if needs_input_grad[1] or needs_input_grad[2]:
    payload, _ = trace_pallas(
        _flash_attention_bwd_dkv,
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        grad_output,
        grad_i,
        block_q_major=min(
            FlashAttention.DEFAULT_BLOCK_SIZES["block_q_major_dkv"],
            q.shape[2]),
        block_k_major=min(
            FlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dkv"],
            k.shape[2]),
        block_k=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_k_dkv"],
                    k.shape[2]),
        block_q=min(FlashAttention.DEFAULT_BLOCK_SIZES["block_q_dkv"],
                    q.shape[2]),
        sm_scale=sm_scale,
        causal=causal,
        mask_value=FlashAttention.DEFAULT_MASK_VALUE,
        debug=False,
        static_argnames=[
            "block_q_major", "block_k_major", "block_k", "block_q", "sm_scale",
            "causal", "mask_value", "debug"
        ],
        use_cache=True)

    grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                 [k.shape, v.shape],
                                                 [k.dtype, v.dtype])

  if needs_input_grad[1]:
    grad_k = grads[0]
  if needs_input_grad[2]:
    grad_v = grads[1]

  # SPMD integration
  if partition_spec is not None:
    grad_q = xs.disable_manual_sharding(
        grad_q, partition_spec, q_full_shape, mesh=mesh).global_tensor
    grad_k = xs.disable_manual_sharding(
        grad_k, partition_spec, kv_full_shape, mesh=mesh).global_tensor
    grad_v = xs.disable_manual_sharding(
        grad_v, partition_spec, kv_full_shape, mesh=mesh).global_tensor

  assert partition_spec is not None

  return grad_q, grad_k, grad_v


@fa_custom_backward.register_fake
def fa_custom_backward_fake(grad_output, q, k, v, o, l, m, q_shape, k_shape):
  if _DEBUG:
    print("Inside fake fa_custom_backward")
  return torch.empty_like(grad_output), torch.empty_like(
      grad_output), torch.empty_like(grad_output)


class FlashAttention2(torch.autograd.Function):

  @staticmethod
  def forward(ctx, q, k, v):
    with torch.no_grad():
      ctx.q_shape = q.shape
      ctx.k_shape = k.shape

      outs = torch.ops.xla.fa_custom_forward(q, k, v)
      if _DEBUG:
        print("forward done with fa_custom_forward")

      o = outs[0]
      full_q, full_k, full_v, l, m = [x for x in outs[1:]]

      # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
      # but it should be OK as the backward will use the same partition_spec
      ctx.save_for_backward(full_q, full_k, full_v, o, l, m)
      return o

  @staticmethod
  def backward(ctx, grad_output):
    with torch.no_grad():
      grad_ab = None
      if _DEBUG:
        print("Inside backward")

      saved = [v for v in ctx.saved_tensors]
      if _DEBUG:
        for t in [grad_output] + saved:
          describe_value(t)

      return torch.ops.xla.fa_custom_backward(grad_output, *saved,
                                              list(ctx.q_shape),
                                              list(ctx.k_shape))


def flash_attention_2(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    q_segment_ids=None,  # [batch_size, q_seq_len]
    kv_segment_ids=None,
    sm_scale=1.0,
    *,
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    partition_spec=None,
    mesh=None,
):
  assert causal, "causal must be True"
  assert partition_spec == ('fsdp', 'tensor', None, None)
  return FlashAttention2.apply(q, k, v)
