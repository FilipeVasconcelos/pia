#coding : utf-8

from RNmod import reseau 

if __name__ == "__main__" :

    nn  = [2,3,1]

    RN = reseau.MCP(nn,verbeux=2)

    X0 = [ [ 0., 0. ] ,
           [ 0., 1. ] ,
           [ 1., 0. ] ,
           [ 1., 1. ] ]

    T  =   [ [ 0. ] ,
             [ 1. ] ,
             [ 1. ] ,
             [ 0. ] ]
    apprentissage = X0, T
    evaluation=[]

    RN.gradient_descent( apprentissage, 100000, 1.0, evaluation )


