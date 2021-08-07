import numpy as np

def softmax(a):
    a = np.array(a)
    a = np.exp(a - a.max())
    return a / a.sum()

