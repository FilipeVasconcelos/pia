 
import numpy as np
import sys 
from mnist import read, show, ascii_show

def sigmoid (x): return 1/(1 + np.exp(-x))        # activation function
def sigmoid_(x): return x * (1 - x)               # derivative of sigmoid
def vinput(x)  : 
    a = np.zeros(10)
    a[x]=1
    return a


epochs = 20000                                  # Number of iterations
chunk  = 60000  
inputLayerSize, hiddenLayerSize1, hiddenLayerSize2, outputLayerSize = 784, 16, 16, 10 
Wh1 = (2.*np.random.uniform(size=(inputLayerSize  , hiddenLayerSize1))-1.)*0.5 
Wh2 = (2.*np.random.uniform(size=(hiddenLayerSize1, hiddenLayerSize2))-1.)*0.5
Wz  = (2.*np.random.uniform(size=(hiddenLayerSize2, outputLayerSize ))-1.)*0.5 
L = 0.001                                            # learning rate      

X=np.empty(shape=(1,inputLayerSize))
Y=np.empty(shape=(1,outputLayerSize))
print("lecture de mnist")
mnist = read()
c = 0
while c < chunk :
    image = next(mnist)
    print(60*"=")
    print(20*" "+"image label",image[0])
    print(60*"=")
    ascii_show(image[1])
    a = np.reshape(image[1], (1,np.product(image[1].shape))) 
    b = np.reshape(vinput(image[0]),(1,np.product(vinput(image[0]).shape)))
    X = np.concatenate ( [ X ,  a ] )
    Y = np.concatenate ( [ Y ,  b ] )
    c +=1
    print(c)

X=np.delete(X, 0, 0)
Y=np.delete(Y, 0, 0)
#X = X - 128
#X = X * 0.00392156862745098
print (X.shape)
print (Y.shape)
print(X[0])
print(Y[0])
print("ok")
for i in range(epochs):
 
    H1   = sigmoid(np.dot(X, Wh1))                  # hidden layer results
    H2   = sigmoid(np.dot(H1,Wh2))                  # hidden layer results
    Z    = sigmoid(np.dot(H2,Wz) )                 # output layer
#    Z    = np.dot(H2,Wz)                            # output layer
#    if i == 0 : print("premier passage\n",Z)
    #print(H1.shape)
    #print(H2.shape)
    #print(Z.shape)
    #sys.exit()
    E    = Y - Z                                   # how much we missed (error)
    if i%1 == 0 :
        print(np.dot(X, Wh1)[0])                  # hidden layer results
        print(np.dot(H1,Wh2)[0])                  # hidden layer results
        print(np.dot(H2,Wz)[0] )                 # output layer
        print(H1[0])
        print(H2[0])
        print(Z[0])
        print(Y[0])
        print(i,(E**2).sum()/chunk)
    dZ   = L * E * sigmoid_(Z)                         # delta Z
    dH2  = L * np.dot( dZ , Wz.T)  * sigmoid_(H2)           # delta H2
    dH1  = L * np.dot(dH2, Wh2.T) * sigmoid_(H1)
    Wz  += np.dot(H2.T,dZ)
    Wh2 += np.dot(H1.T,dH2)
    Wh1 += np.dot(X.T,dH1)

for i,e in enumerate(Z) :
    print(Y[i],e)




