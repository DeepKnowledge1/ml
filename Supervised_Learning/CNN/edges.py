import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess image
img_path = r'C:\Users\abdulgader\Downloads\img1.jpg'  # replace with your image path
img = Image.open(img_path).convert('L')  # convert to grayscale
transform = T.ToTensor()
img_tensor = transform(img).unsqueeze(0)  # shape: [1, 1, H, W]

# Define vertical and horizontal edge detection kernels
kernel_vertical = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)  # shape: [1, 1, 3, 3]

kernel_horizontal = torch.tensor([[[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)  # shape: [1, 1, 3, 3]

# Apply convolution
edge_vertical = F.conv2d(img_tensor, kernel_vertical, padding=1)
edge_horizontal = F.conv2d(img_tensor, kernel_horizontal, padding=1)

# Show results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(img_tensor.squeeze(), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Vertical Edges')
plt.imshow(edge_vertical.squeeze().abs(), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Horizontal Edges')
plt.imshow(edge_horizontal.squeeze().abs(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
