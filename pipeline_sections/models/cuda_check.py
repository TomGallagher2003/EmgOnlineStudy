"""Quick PyTorch/CUDA environment probe.

Prints the installed PyTorch version, the compiled CUDA toolkit version that
PyTorch was built against, whether CUDA is available at runtime, and the name
of CUDA device 0.

Notes:
    - ``torch.version.cuda`` reports the CUDA toolkit version PyTorch was built
      with, which may differ from the driver/toolkit installed on your system.
    - ``torch.cuda.get_device_name(0)`` assumes a CUDA device is present and
      visible as index 0.
"""

import torch

print("Torch Version ->", torch.__version__)
print("Cuda Version ->", torch.version.cuda)
print("Cuda Is Available ->", torch.cuda.is_available())
print("Cuda Device ->", torch.cuda.get_device_name(0))
