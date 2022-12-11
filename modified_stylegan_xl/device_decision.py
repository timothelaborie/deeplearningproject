import torch
torch.set_default_dtype(torch.float32)

# Get cpu or gpu device for training.
if torch.cuda.is_available():
    device = 'cuda:0'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'
print(f"Device used: {device}")