import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 2 inputs → 2 neurons
        self.fc2 = nn.Linear(2, 1)  # 2 neurons → 1 output

    def forward(self, x):
        x = F.relu(self.fc1(x))     # Hidden layer
        x = torch.sigmoid(self.fc2(x))  # Output layer
        return x


X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

model = SimpleNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
