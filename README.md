# proj_mucep

Hybrid ODE-CNN Model:

The HybridODECNN class defines a CNN model that incorporates ODE layers for image classification tasks.
Load CIFAR-10, train for 10 epochs, and evaluate the model's accuracy using train(epochs) and evaluate() functions.
Hybrid GD-ODE Optimizer:

A custom optimizer class HybridGDODE is defined, which dynamically switches between GD and ODE-based optimization.
It includes a test case where the optimizer is applied to optimize the Rosenbrock function, a classic optimization benchmark.
