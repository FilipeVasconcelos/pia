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


    nn  = [img_taille[0]*img_taille[1],16,10] 
        
    X0 = []
    T  = []
    apprentissage = mnist.charger_donnees()
    print ( "lecture des donnees mnist" ) 

    #print( len (apprentissage[0]) , len (apprentissage[1]) )
    napp = len (apprentissage[0])
    kb= 7500 
    nb = int ( napp / kb ) 
    #print(napp,kb,nb)
    evaluation=[]
    W=None
    B=None
    for k in range(kb):
        print("batch : ", k,"/",kb)
        #print ( len( apprentissage[0][k*nb:((k+1)*nb-1)]) ) 
        #print ( k*nb,(k+1)*nb-1 ) 
        X0 = apprentissage[0][k*nb:(k+1)*nb-1]
        T  = apprentissage[1][k*nb:(k+1)*nb-1]
        batch = X0, T 
        RN = reseau.MCP(nn,verbeux=2,verbe_periode=5000,sigma=1.0,mu=0.,W=W,B=B)
        W, B = RN.gradient_descent( batch, 5000, 1.0, evaluation )
    sys.exit()
    
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1,sigma=1.0,mu=0.)
    W, B = RN.gradient_descent( apprentissage, 6000, 0.0001, evaluation )

    sys.exit()

    neval = 1000
    tailles=[3,5,7]
    imgs = []
    lbls = []
    # générer les labels parmis tailles
    for i in range(neval) :
        lbls.append(random.choice(tailles))
    #parmis les labels on génère les images
    for lbl in lbls:
        img_gen = gen_carre(img_taille, lbl)
        X0.append( np.reshape (img_gen,(nn[0]) ) )
        T.append(resultat_vecteur(lbl))
    
    evaluation= X0,T
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1000,sigma=0.1,mu=0.,W=W,B=B)
    Y = RN.gradient_descent( apprentissage, 10000, 0.01, evaluation )
    
    #print(Y)
    #sys.exit()        
    cok = 0
    cal = 0
    for i,t in enumerate(T):
        #print(t,Y[i])
        vt = np.argmax(t)
        ve = np.argmax(Y[i])
        if vt == ve:
            cok += 1
        cal += 1
    print(cal,cok)
    print()
    print()
    print("evaluation score : {:6.2f} %".format((cok/neval)*100.))
 
















