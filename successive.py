'''
    This neural network takes a number as input and returns 
    the successive number as output.
    For example, with four numbers:
    input ( number 2 ): [0, 1, 0, 0]
    output (number 3 ): [0, 0, 1, 0]
'''

from MyNetwork.neuron import sigmoid, heaviside
from MyNetwork.network import Network
import random

numbers = 6

print(f'\nLearning... ({numbers} numbers)\n')

network = Network(numbers, ((numbers, sigmoid), (numbers, sigmoid)))

learning_iterations = 10000
for i in range(learning_iterations):
    input_ = [0 for _ in range(numbers)]
    target_output = input_.copy()

    value = random.randint(0, len(input_)-2)
    input_[value] = 1
    target_output[value+1] = 1

    network.learn(input_, 0.8, target_output)

for i in range(numbers):
    input_ = [0 for _ in range(numbers)]
    input_[i] = 1
    res = [int(round(x, 0)) for x in network.run(input_)]
    print(input_, res, sep=' -> ')


