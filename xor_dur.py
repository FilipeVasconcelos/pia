#coding : utf-8

from RNmod import reseau 
import numpy as np
import sys

if __name__ == "__main__" :

    nn  = [2,2,1]

    #RN = reseau.MCP(nn,verbeux=12,verbe_periode=1000,distrib_poids="uniforme",activation="binary_step")

    #paramètres optimisées
    W=[np.array([[ 6.8900000 ,  8.5958971 ] ,
                 [ 6.8900000 ,  8.5945971]]), np.array([[-18.8870000],[ 18.88870000]])]
    B=[np.array([[-10.50616154,  -4.01803533]]), np.array([[-8.70920879]])]
    #W=[np.array([[ 1.0 , -1.0 ], [ -1.0 ,  1.0]]),np.array([[1.0],[1.0]])]
    #B=[np.array([[-1.0,-1.0]]),np.array([[-0.1]])]


    print(W)
    print(B)
    #sys.exit()

    X0 = [ [ 0., 0. ] ,
           [ 0., 1. ] ,
           [ 1., 0. ] ,
           [ 1., 1. ] ]

    T  =   [ [ 0. ] ,
             [ 1. ] ,
             [ 1. ] ,
             [ 0. ] ]
    apprentissage = X0,T
    evaluation = X0, T 
    RN = reseau.MCP(nn,verbeux=12,verbe_periode=10000,W=W,B=B,activation="binary_step")
#    RN = reseau.MCP(nn,verbeux=12,verbe_periode=10000,W=W,B=B,activation="sigmoide")
    Y = RN.gradient_descent( apprentissage, 100000, 1.0, evaluation )

    cok = 0
    for i,t in enumerate(T):
        print(t,Y[i])
        vt = np.argmax(t)
        ve = np.argmax(Y[i])
        if vt == ve:
            cok += 1
    print(cok)
    print()
    print()
    print("evaluation score : {:6.2f} %".format((cok/4.)*100.))

