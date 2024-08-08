import pytest
import torch, torch_xla
import torch_xla.core.xla_model as xm
import numpy as np


def run_test(dtype, uplo):
  # Generate a complex Hermitian matrix.
  cpu = torch.device("cpu")
  A = torch.randn(2, 2, dtype=dtype, device=cpu)
  A = (A + A.T.conj()) / 2
  for i in range(A.shape[-1]):
    A[i, i].imag.zero_()
  assert np.allclose(A.conj().T.resolve_conj().numpy(), A.numpy())

  print("Build LazyTensor IR")
  A = A.clone().to(torch_xla.device())
  L, V = torch.linalg.eigh(A, UPLO=uplo)

  print("L.dtype = ", L.dtype)
  print("V.dtype = ", V.dtype)

  import torch_xla.debug.metrics as met
  print(met.metrics_report())

  print("Print IR and lowered graph")
  print(torch_xla._XLAC._get_xla_tensors_text([L, V]))
  print(torch_xla._XLAC._get_xla_tensors_hlo([L, V]))

  print("Eval")
  xm.mark_step()
  print("L = ", L, L.dtype)
  print("V = ", V, V.dtype)

  print("Compare with CPU")
  A_CPU = A.cpu()
  L_CPU, V_CPU = torch.linalg.eigh(A_CPU, UPLO=uplo)
  print("L = ", L_CPU, L_CPU.dtype)
  print("V = ", V_CPU, V_CPU.dtype)

  assert L.dtype == L_CPU.dtype
  assert V.dtype == V_CPU.dtype

  # Eigenvalues should be close.
  assert np.allclose(L.cpu().numpy(), L_CPU.numpy())

  # The eigenvectors of a symmetric matrix are not unique,
  # nor are they continuous with respect to A.
  # Due to this lack of uniqueness, different hardware and
  # software may compute different eigenvectors.
  assert np.allclose(
      (V_CPU @ torch.diag(L_CPU).type(torch.complex64)
       @ V_CPU.T.conj()).numpy(),
      A_CPU.numpy(),
      rtol=1e-2,
      atol=1e-3), "Torch CPU failed"
  x = (V @ torch.diag(L).type(torch.complex64) @ V.T.conj()).cpu().numpy()
  y = A.cpu().numpy()
  assert np.allclose(x, y, rtol=1e-2, atol=1e-3), f"XLA failed: {x - y}"


@pytest.mark.parametrize("i", list(range(100)))
@pytest.mark.parametrize("uplo", ['U', 'L'])
def test_eval(i, uplo):
  run_test(torch.complex64, uplo)
