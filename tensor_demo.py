import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print(x + y)
print(x - y)
print(x * y)
print(torch.dot(x, y))


a= torch.zeros((2, 3))
b = torch.ones((2, 3))
c = torch.rand(2, 3)
print(a)
print(b)
print(c)

print(a.shape)
print(b.shape)
print(c.shape)

print(a.dtype)
print(b.dtype)
print(c.dtype)