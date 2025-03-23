'''
    This neural network takes 2 bits as input
    and gives the output converted in decimal.
    For example:
        input (0b10): [1, 0]
        output (2): [0, 0, 1, 0]
'''

from MyNetwork.neuron import sigmoid
from MyNetwork.network import Network
import random

network = Network(2, ((4, sigmoid), (4, sigmoid)))

print('\nLearning...\n')
for i in range(10000):
    x = random.choice([0, 1])
    y = random.choice([0, 1])

    res = int(f'{x}{y}', 2)
    target_output = [0, 0, 0, 0]
    target_output[res] = 1

    network.learn((x, y), 0.8, target_output)

print('BINARY | OUTPUT')
for i in range(4):
    b = [int(x) for x in bin(i)[2:].zfill(2)]
    res = [int(round(x, 0)) for x in network.run(b)]
    r = res.index(max(res))
    print(b, '|', f'({r})', res)

