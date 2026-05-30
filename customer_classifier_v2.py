import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.array([
    [2, 1, 100, 0],
    [10, 5, 500, 1],
    [8, 4, 300, 1],
    [1, 1, 50, 0],
    [7, 6, 400, 1],
    [3, 2, 150, 0],
    [12, 8, 700, 1],
    [0.5, 1, 3, 0],
    [9, 7, 600, 1],
    [4, 3, 200, 0],
])

y = np.array([0, 2, 1, 0, 2, 0, 2, 0, 2, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).clone().detach()
X_test = torch.tensor(X_test, dtype=torch.float32).clone().detach()
y_train = torch.tensor(y_train, dtype=torch.long).clone().detach()
y_test = torch.tensor(y_test, dtype=torch.long).clone().detach()

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):        
        return self.net(x)


model = Model()                  
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy}")