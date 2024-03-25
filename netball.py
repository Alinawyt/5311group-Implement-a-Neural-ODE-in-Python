import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import time
import sys
import gc

# Load data
data = pd.read_csv("./netball.csv")
x0, y0, v0, theta0 = data.loc[0, ['x', 'y', 'v', 'theta']]

# Define neural network model
class dudt_model(nn.Module):
    def __init__(self):
        super(dudt_model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

    def forward(self, t, y):
        return self.net(y)

# Define loss function
def loss_fn(pred_traj, true_traj):
    return torch.sum((pred_traj - true_traj) ** 2)

# Define initial conditions
u0 = torch.tensor([x0, y0, v0 * np.cos(np.radians(theta0)), v0 * np.sin(np.radians(theta0))], dtype=torch.float32)

# Prepare true trajectory
true_traj = torch.tensor(data[['x', 'y']].values, dtype=torch.float32)

# Define model
model = dudt_model()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    pred_traj = odeint(model, u0[:2], torch.tensor(data['t'].values, dtype=torch.float32))
    loss = loss_fn(pred_traj, true_traj)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
print(f"Epoch {epochs}: Loss = {loss.item()}")
# Predict trajectory using Neural ODEs
# Get current memory usage
start_memory = gc.get_objects()
# Get current time
start_time = time.time()
pred_traj = odeint(model, u0[:2], torch.tensor(data['t'].values, dtype=torch.float32))
end_time = time.time()
end_memory = gc.get_objects()
execution_time = end_time - start_time
print("Execution time for Python(torchdiffeq.odeint) implememt Neural ODEs:\n", execution_time, "seconds")
# Calculate the increment of memory allocation
allocations = len(end_memory) - len(start_memory)
# Calculate the size of memory allocation
memory_allocation = sum(sys.getsizeof(obj) for obj in end_memory) - sum(sys.getsizeof(obj) for obj in start_memory)
print(f"{allocations} allocations: {memory_allocation / 1024:.3f} KiB")

# Plot trajectories
plt.plot(pred_traj[:, 0].detach().numpy(), pred_traj[:, 1].detach().numpy(), label="Predicted trajectory")
plt.plot(data['x'], data['y'], label="True trajectory")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Netball trajectory (Implementing with Python)')
plt.legend()
plt.show()