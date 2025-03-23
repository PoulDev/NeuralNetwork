import random
from .neuron import HidNeuron, OutNeuron

class Network:
    def __init__(self, nInputs, layers):
        '''
            layers: a tuple containing (nNeurons, activationFunction) for every layer
        '''
        self.network = [[] for _ in layers]

        nWeights = nInputs
        for i, layer in enumerate(layers):
            Neuron = OutNeuron if (i == len(layers)-1) else HidNeuron
            function = layer[1]

            for _ in range(layer[0]):
                weights = [random.random() for _ in range(nWeights)]
                self.network[i].append(Neuron(function, random.random(), *weights))

            nWeights = layer[0]


    def run(self, inputs):
        values = inputs
        for layer in self.network:
            layer_res = []
            for neuron in layer:
                neuron.run(*values)
                layer_res.append(neuron.output)
            values = layer_res

        return values

    def learn(self, inputs, lr, target_output):
        self.run(inputs)

        # Calculate the deltas for the output neurons
        for tvalue, neuron in zip(target_output, self.network[-1]):
            neuron.updateDelta(tvalue)

        # Calculate the deltas for all the hidden layers
        for index in range(len(self.network)-2, -1, -1):
            layer = self.network[index]
            for nindex, neuron in enumerate(layer):
                nl_deltas = [n.delta for n in self.network[index+1]]
                nl_weights = [n.weights[nindex] for n in self.network[index+1]]
                neuron.updateDelta(nl_deltas, nl_weights)

        # Update the weights & biases
        for ila, layer in enumerate(self.network):
            for neuron in layer:
                for iwe in range(len(neuron.weights)):
                    if ila-1 >= 0:
                        output = self.network[ila-1][iwe].output
                    else:
                        output = inputs[iwe]
                    
                    neuron.weights[iwe] += lr * neuron.delta * output
                    neuron.bias += lr * neuron.delta

