import numpy as np
def sigmoide (x): return 1/(1 + np.exp(-x))        # activation function
def sigmoide_(x): return x * (1 - x)               # derivative of sigmoid
