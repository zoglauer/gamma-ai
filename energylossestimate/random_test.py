import numpy as np
import numpy.linalg as LA
import scipy
from scipy.special import gamma

arr = np.array([[1, 2, 3], [1, 2, 3]])
rr = np.array([[0, 0, 0], [0, 0, 0]])
print(arr.shape)
print(rr.shape)
c = arr-rr
print(c.shape)
d1 = LA.norm(arr, axis=1, ord=2)
d2 = LA.norm(c, axis=1, ord=2)
print(d1.shape, d1)
print(d2.shape, d2)
print(arr**2)

def f(beta, distance, alpha):
    g = gamma(alpha)
    num = (beta * distance)**(alpha - 1) * beta * np.exp(-1 * beta * distance)
    return distance * (num / g)

print(f(1, d1, 1))
