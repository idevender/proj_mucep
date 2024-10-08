# Hybrid ODE-CNN Model and Hybrid GD-ODE Optimizer Implementation

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint as odeint
import math
import matplotlib.pyplot as plt

# ------------------------------- 1. Hybrid ODE-CNN Model -------------------------------

# Define the ODE function for CNN layers


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out

# Define the ODE block using ODEFunc


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=1e-3, atol=1e-3)
        return out[1]

# Define the Hybrid ODE-CNN model


class HybridODECNN(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridODECNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.odeblock = ODEBlock(ODEFunc(64))

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.maxpool(out)

        out = self.odeblock(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Data loading and preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Initialize the Hybrid ODE-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridODECNN().to(device)

# Loss function and optimizer for Hybrid ODE-CNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for Hybrid ODE-CNN model


def train(epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

# Evaluation function for Hybrid ODE-CNN model


def evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f}%')


# Train and evaluate the Hybrid ODE-CNN model
train(10)
evaluate()

# ----------------------------- 2. Hybrid GD-ODE Optimizer -----------------------------

# Define the Hybrid GD-ODE optimizer


class HybridGDODE(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, switching_threshold=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, buffer=None,
                        beta1=beta1, beta2=beta2, eps=eps)
        super(HybridGDODE, self).__init__(params, defaults)
        self.switching_threshold = switching_threshold

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1

                if torch.norm(grad) < self.switching_threshold:
                    # Use ODE-based optimization
                    def ode_func(t, x):
                        return -grad

                    integration_time = torch.tensor([0, group['lr']])
                    solution = odeint(ode_func, p.data,
                                      integration_time, method='rk4')
                    p.data = solution[-1]
                else:
                    # Use Adam-like optimization
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['beta1'], group['beta2']

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    step_size = group['lr'] * math.sqrt(
                        1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# Example usage of Hybrid GD-ODE Optimizer

# Define a simple Rosenbrock function model for optimization


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.x = torch.nn.Parameter(torch.tensor([-1.0, 1.0]))

    def forward(self):
        return rosenbrock(self.x[0], self.x[1])


# Initialize the simple model and Hybrid GD-ODE optimizer
model = SimpleModel()
optimizer = HybridGDODE(model.parameters(), lr=0.01)

# Training loop for optimizing the Rosenbrock function
n_iterations = 1000
for i in range(n_iterations):
    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    if i % 100 == 0:
        print(f'Iteration {i}, Loss: {loss.item():.4f}, x: {model.x.data}')

print(f'Final x: {model.x.data}, Final loss: {loss.item():.4f}')
