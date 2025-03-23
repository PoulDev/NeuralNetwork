import math

step_function = lambda x: x / abs(x)
sigmoid = lambda x: 1 / (1 + math.exp(-x))
heaviside = lambda x: (int(x > 0) if x != 0 else 0.5)

#sigmoid_dvt = lambda x: sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, function, bias, *weights):
        self.weights = list(weights)
        self.bias: float = bias
        self.af = function
        
        # Learning Variables
        self.output = 0
        self.delta  = 0

    def run(self, *inputs):
        self.output = self.af(self._run(*inputs))

    def _run(self, *inputs):
        total = self.bias
        for input, weight in zip(inputs, self.weights):
            total += input * weight

        return total

    def __str__(self):
        return f'Neuron(b={self.bias}, w={self.weights})'

    def __repr__(self):
        return str(self)

class OutNeuron(Neuron): # Output Neurons
    def updateDelta(self, target):
        self.delta = self.output * (1 - self.output) * (target - self.output)

class HidNeuron(Neuron): # Hidden Neurons
    def updateDelta(self, nl_deltas, nl_weights):
        deltasum = sum([w * o for w, o in zip(nl_weights, nl_deltas)])
        self.delta = self.output * (1 - self.output) * deltasum


