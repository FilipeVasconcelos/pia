#   XOR.py-A very simple neural network to do exclusive or.
#   sigmoid activation for hidden layer, no (or linear) activation for output
 
import numpy as np
import sys 
from mnist import read

def sigmoid (x): return 1/(1 + np.exp(-x))        # activation function
def sigmoid_(x): return x * (1 - x)               # derivative of sigmoid
def vinput(x)  : 
    a = np.zeros(10)
    a[x]=1
    return a


epochs = 100                                   # Number of iterations
inputLayerSize, hiddenLayerSize1, hiddenLayerSize2, outputLayerSize = 784, 16, 16, 10 
L = .22                                            # learning rate      

mnist = read()
while True :
    image = next(mnist)
    X =  np.reshape(image[1], (1,np.product(image[1].shape)))
    Y = np.reshape(vinput(image[0]),(1,np.product(vinput(image[0]).shape)))
                                                  # weights on layer inputs
    Wh1 = np.random.uniform(size=(inputLayerSize  , hiddenLayerSize1))
    Wh2 = np.random.uniform(size=(hiddenLayerSize1, hiddenLayerSize2))
    Wz  = np.random.uniform(size=(hiddenLayerSize2, outputLayerSize ))

    #print(Wh1.shape)
    #print(Wh2.shape)
    #print(Wz.shape)
    #print(X.shape)
    #print(Y.T)
    for i in range(epochs):
 
        H1   = sigmoid(np.dot(X, Wh1))                  # hidden layer results
        H2   = sigmoid(np.dot(H1,Wh2))                  # hidden layer results
        Z    = sigmoid(np.dot(H2,Wz) )                 # output layer
        #print(H1.shape)
        #print(H2.shape)
        #print(Z.shape)
        #sys.exit()
        E    = Y - Z                                   # how much we missed (error)
        #print(E)
        #print(Z)
        dZ   = E * sigmoid_(Z)                         # delta Z
        dH2  = np.dot( dZ , Wz.T)  * sigmoid_(H2)           # delta H2
        dH1  = np.dot(dH2, Wh2.T) * sigmoid_(H1)
        Wz  += np.dot(H2.T,dZ)
        Wh2 += np.dot(H1.T,dH2)
        Wh1 += np.dot(X.T,dH1)
        print((E**2).sum())
        #print(Wz.shape,H2.T.shape,H2.shape,dZ.shape) 


#    dZ = E * sigmoid_(Z)                        # delta Z
#    dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
#    Wz +=  H.T.dot(dZ)                          # update output layer weights
#    Wh +=  X.T.dot(dH)    



#    print(Wh)
#    print(Wz)
#    print(Z)                # what have we learnt?
