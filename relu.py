import numpy as np

def relu(x):
    return np.maximum(0, x)

values = np.array([-5, -2, -1, 0, 1, 2, 3, 7])

print("before relu:", values)
print("after relu: ", relu(values))