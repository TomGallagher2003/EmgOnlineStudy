import torch
print("Torch Version ->", torch.__version__)
print("Cuda Version ->", torch.version.cuda)
print("Cuda Is Available ->", torch.cuda.is_available())
print("Cuda Device ->", torch.cuda.get_device_name(0))
