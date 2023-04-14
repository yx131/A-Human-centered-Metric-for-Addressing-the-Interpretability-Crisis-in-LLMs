import torch

gpu_count = torch.cuda.device_count()
#gpu_device = f'cuda:{gpu_count-1}' if gpu_count > 0 else 'cpu'
gpu_device = f'cuda:0' if gpu_count > 0 else 'cpu'
