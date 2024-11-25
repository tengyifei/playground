import torch
cpu = torch.device("cpu")
A = torch.randn(2, 2, dtype=torch.complex64, device=cpu)
L, V = torch.linalg.eigh(A, UPLO='L')
print(L, V)
