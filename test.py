import torch

# Assuming x is your tensor of shape (100, 20)
x = torch.randn(100, 20) # Example tensor

# Calculate the mean of the second dimension
mean = x.mean(dim=0, keepdim=True)

print(mean.shape)