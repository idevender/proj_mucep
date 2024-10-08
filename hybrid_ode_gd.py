import torch
from torch.optim import Optimizer
from torchdiffeq import odeint_adjoint as odeint
import math


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

# Example usage


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.x = torch.nn.Parameter(torch.tensor([-1.0, 1.0]))

    def forward(self):
        return rosenbrock(self.x[0], self.x[1])


# Initialize the model and optimizer
model = SimpleModel()
optimizer = HybridGDODE(model.parameters(), lr=0.01)

# Training loop
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
