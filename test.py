import torch

# Example tensor of shape (10, 5, 5)
# Replace this with your actual tensor
tensor = torch.randn(10, 5, 5)

# Setting some slices to all zeros for demonstration purposes
# Remove or modify these lines according to your actual data
tensor[2, :, :] = 0
tensor[7, :, :] = 0

# Find indices where all values in the 5x5 matrices are zeros
# Apply 'all' method across both dimensions separately
zero_indices = tensor == 0
zero_indices = zero_indices.all(dim=2).all(dim=1)

# Extract the indices
zero_indices = zero_indices.nonzero(as_tuple=False).squeeze()

# Print the indices
print(zero_indices)
