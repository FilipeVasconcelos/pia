#coding : utf-8

from RNmod import reseau 
import numpy as np
import sys

if __name__ == "__main__" :

    nn  = [2,2,1,1]

    RN = reseau.MCP(nn,verbeux=3,verbe_periode=1000,distrib_poids="uniforme")

    W=[np.array([[1.0,-1.0],[1.0,-1.0]]),np.array([[1.0],[1.0]]),np.array([[1.0]])]
    B=[np.array([[0.5,-1.5]]),np.array([[1.5]]),np.array([[0.0]])]

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
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=10000,W=W,B=B)
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

