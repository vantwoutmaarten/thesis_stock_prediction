import numpy as np

arr = np.random.random((5,5))

print(arr)

x = -arr[:2]
print("and now the slice")

print(x)