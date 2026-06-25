from enum import IntEnum
import numpy as np
import torch
class Shit(IntEnum):
    SHIT1 = 0
    SHIT2 = 1
    SHIT3 = 2
    SHIT4 = 3

num_agents = 4

hp = np.zeros(num_agents, dtype = np.float32)

hp = np.array([2.4, 34.5, 53, 0])
boolean = np.array([True, False, True , True])

print(hp > 0)
print(torch.device("cpu"), isinstance(torch.device("cpu"), str), type(torch.device("cpu")))
print(np.dtype(np.bool_).itemsize)

gol = 2
mol = 3
jhol = 4
if boolean:
    print("yes")

