import torch
import torch.nn as nn
import torch.optim as optim

# inputs: [feature1, feature2]
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

# labels
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
    def forward(self, x):
        return self.net(x)

model = BinaryClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logits = model(X)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

print("Probabilities:")
print(probs)
print("Predictions:")
print(preds)