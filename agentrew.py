from enum import IntEnum
import numpy as np
class Shit(IntEnum):
    SHIT1 = 0
    SHIT2 = 1
    SHIT3 = 2
    SHIT4 = 3


shit_idx = 0
#shit_value = Shit(shit_idx)
role_idx = np.zeros(4, dtype = float)
ga = False
#role_idx[shit_value] = 1
print(role_idx)

if shit_idx in Shit:
    ga = True
    print(ga)

print(ga)

role_idx[0] = 21
role_idx[1] = 2
role_idx[2] = 5
role_idx[3] = 58

while 5 in role_idx:
    print("id")
    break

array = np.zeros(4, dtype = bool)
array[3] = True 
print(role_idx[array])
print(type(Shit.SHIT1))
print(isinstance(Shit.SHIT1, int))