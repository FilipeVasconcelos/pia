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
    #              FNN !!!
    # =================================
    ds_mnist = mnist.lire_ds()


    nn  = [img_taille[0]*img_taille[1],16,16,3] 
        
    X0 = []
    T  = []
    napp = 60000 
    for i in range(napp) :
        image = next(ds_mnist)
        #mnist.ascii_show(image[1])
        mnist.grayramp_show ( image[1] )
        time.sleep(1.0)

    sys.exit()
    apprentissage = X0, T  
    evaluation=[]
    
    X0 = []
    T  = []

    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1000,sigma=0.1,mu=0.)
    W, B = RN.gradient_descent( apprentissage, 40000, 0.01, evaluation )


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
    
    print(Y)
    #sys.exit()        
    cok = 0
    cal = 0
    for i,t in enumerate(T):
        print(t,Y[i])
        vt = np.argmax(t)
        ve = np.argmax(Y[i])
        if vt == ve:
            cok += 1
        cal += 1
    print(cal,cok)
    print()
    print()
    print("evaluation score : {:6.2f} %".format((cok/neval)*100.))
 
















