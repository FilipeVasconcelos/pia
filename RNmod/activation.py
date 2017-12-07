import numpy as np
def sigmoide (x): return 1/(1 + np.exp(-x))        # activation function
def sigmoide_(x): return x * (1 - x)               # derivative of sigmoid
def binary_step(x): 
    out=np.asarray(x)
    for i,v in enumerate(x) :
        for j,vi in enumerate(v):
            if vi < 0. : 
                out[i][j]=0.
            else: 
                out[i][j]=1.
    return out

if __name__ == "__main__":

    print("test bloc du module activation")
