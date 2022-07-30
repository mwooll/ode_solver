import numpy as np
def a(X):
    if hasattr(X, "__iter__"):
        return np.array([a(x) for x in X])
    return 1

def f(X):


    return np.sin(X*np.pi)

def g(X):


    return np.sin(X*np.pi)/np.pi**2
