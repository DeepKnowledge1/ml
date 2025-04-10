import torch
import torch.nn as nn

# Input image: Batch size 1, 3 channels (RGB), 32x32 pixels
input_image = torch.randn(1, 3, 32, 32)

# Define convolutional layer: 3 input channels, 32 filters, 3x3 kernel
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=0  # No padding
)

# Pass input through the conv layer
output = conv_layer(input_image)

# Show the output shape
print("Output shape:", output.shape)

# Output shape: torch.Size([ 30, 30])