def sigmoid (x): return 1/(1 + np.exp(-x))        # activation function
def sigmoid_(x): return x * (1 - x)               # derivative of sigmoid
