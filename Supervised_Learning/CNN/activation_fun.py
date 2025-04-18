import torch
import torch.nn as nn

# Simulated CNN feature map (as seen in your image)
# Shape: (1, 1, 3, 3) → (batch_size, channels, height, width)
feature_map = torch.tensor([[[[5.0, 13.0, 0.0],
                              [13.0, 12.0, 4.0],
                              [0.0, 15.0, 12.0]]]])

# Define sigmoid activation
sigmoid = nn.Sigmoid()

# Apply sigmoid
activated_map = sigmoid(feature_map)

# Print original and activated maps
print("🔹 Original Feature Map:\n", feature_map)
print("\n🔹 After Sigmoid Activation:\n", activated_map)
