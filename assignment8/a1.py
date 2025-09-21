import numpy as np

# Summation Unit
def summation_unit(inputs, weights, bias=0):
    return np.dot(inputs, weights) + bias

# Activation Functions
def step_activation(x): return 1 if x >= 0 else 0
def bipolar_step_activation(x): return 1 if x >= 0 else -1
def sigmoid_activation(x): return 1 / (1 + np.exp(-x))
def tanh_activation(x): return np.tanh(x)
def relu_activation(x): return max(0, x)
def leaky_relu_activation(x, alpha=0.01): return x if x > 0 else alpha * x

# Comparator
def error_comparator(expected, predicted):
    return expected - predicted

def main():
    inputs = np.array([1, 0])
    weights = np.array([0.5, -0.3])
    bias = 0.1

    net = summation_unit(inputs, weights, bias)
    print("Summation:", net)
    print("Step:", step_activation(net))
    print("Sigmoid:", sigmoid_activation(net))
    print("Error:", error_comparator(1, step_activation(net)))

if __name__ == "__main__":
    main()
