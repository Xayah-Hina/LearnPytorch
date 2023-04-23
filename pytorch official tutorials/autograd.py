import torch

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)  # forward pass
loss = (prediction - labels).sum()
loss.backward()  # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()  # gradient descent

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a ** 3 - b ** 2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)
