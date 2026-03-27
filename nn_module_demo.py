import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
print(model)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])        
y = model(x)
print(y)

# print(model.fc1.weight)
# print(model.fc1.bias)
# print(model.fc2.weight)
# print(model.fc2.bias)

# print(model.fc1.weight.shape)
# print(model.fc1.bias.shape)
# print(model.fc2.weight.shape)
# print(model.fc2.bias.shape)