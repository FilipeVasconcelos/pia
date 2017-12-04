#coding: utf-8

import sys
import time
import numpy as np
import random
from RNmod import mnist
from RNmod import reseau

if __name__ == "__main__":

    img_taille = (28,28)

    # =================================
    #              FNN 
    # =================================
    ds_mnist = mnist.lire_ds()


    nn  = [img_taille[0]*img_taille[1],128,128,10] 
        
    X0 = []
    T  = []
    apprentissage = mnist.charger_donnees(dataset="apprentissage")
    print ( "lecture des donnees mnist" ) 

    #napp = len (apprentissage[0])
    napp=60000
    kb= 1500
    nb = int ( napp / kb ) 
    evaluation=[]
    W=None
    B=None
    for k in range(kb):
        print("batch : ", k,"/",kb)
        X0 = apprentissage[0][k*nb:(k+1)*nb-1]
        T  = apprentissage[1][k*nb:(k+1)*nb-1]
        batch = X0, T 
        RN = reseau.MCP(nn,verbeux=2,verbe_periode=500,distrib_poids="normale",keyg=True,sigma=1.0,mu=0.,W=W,B=B)
        W, B = RN.gradient_descent( batch, 500, 0.1, evaluation )
    
    evaluation = mnist.charger_donnees(dataset= "evaluation")
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1000,distrib_poids="normale",sigma=0.1,mu=0.,W=W,B=B)
    Y = RN.gradient_descent( apprentissage, 10000, 0.1, evaluation )
    
    cok = 0
    cal = 0
    T=evaluation[1]
    neval=len(T)
    for i,t in enumerate(T):
        vt = np.argmax(t)
        ve = np.argmax(Y[i])
        if vt == ve:
            print("OK",vt,ve)
            cok += 1
        else:
            mnist.grayramp_show_(evaluation[0][i])
            print("WRONG",vt,ve)
        cal += 1
    print(cal,cok)
    print("evaluation score : {:6.2f} %".format((cok/neval)*100.))
 
