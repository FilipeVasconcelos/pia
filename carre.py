#coding: utf-8

import sys
import numpy as np
import random
from RNmod import mnist
from RNmod import reseau
# ===================================================================
def resultat_vecteur(e):
    """
    Transforme le label en un vecteur de 10 elements
    """
    if e == 3 : k = 0
    if e == 5 : k = 1
    if e == 7 : k = 2
    v = np.zeros((3))
    v[k] = 1.0
    return v

# ===================================================================
def gen_carre(taille,tkey):
    # coin en bas à gauche
    # tc : taille du carré
    # petit :
    # -1  1 
    img = np.zeros(taille,dtype=int)
    xo =np.random.randint(tkey,28-tkey)
    yo =np.random.randint(tkey,28-tkey)
    #print(" centre du carré ",xo,yo)
    val = 1. 
    if tkey == 3 :
        img[xo+1,yo-1:yo+2 ] = val
        img[xo  ,yo-1:yo+2 ] = val
        img[xo-1,yo-1:yo+2 ] = val
    if tkey == 5 :
        img[xo+2,yo-2:yo+3 ] = val
        img[xo+1,yo-2:yo+3 ] = val
        img[xo  ,yo-2:yo+3 ] = val
        img[xo-1,yo-2:yo+3 ] = val
        img[xo-2,yo-2:yo+3 ] = val
    if tkey == 7 :
        img[xo+3,yo-3:yo+4 ] = val
        img[xo+2,yo-3:yo+4 ] = val
        img[xo+1,yo-3:yo+4 ] = val
        img[xo  ,yo-3:yo+4 ] = val
        img[xo-1,yo-3:yo+4 ] = val
        img[xo-2,yo-3:yo+4 ] = val
        img[xo-3,yo-3:yo+4 ] = val

    return img

if __name__ == "__main__":

    img_taille = (28,28)

    
    # =================================
    # test générations de carrés et 
    # affichage dans la console
    # =================================
    # générer des carrés
    # 3 : petit
    # 5 : moyen
    # 7 : grand
    #img_gen = gen_carre(img_taille, 3)
    #mnist.grayramp_show(img_gen)
    #print(60*"=")
    #img_gen = gen_carre(img_taille, 5)
    #mnist.grayramp_show(img_gen)
    #print(60*"=")
    #img_gen = gen_carre(img_taille, 7)
    #mnist.grayramp_show(img_gen)
    #print(60*"=")


    
    # =================================
    #              FNN !!!
    # =================================

    nn  = [img_taille[0]*img_taille[1],32,16,3] 
        
    X0 = []
    T  = []

    napp = 200 
    tailles=[3,5,7]
    imgs = []
    lbls = []
    # générer les labels parmis tailles
    for i in range(napp) :
        lbls.append(random.choice(tailles))
    #parmis les labels on génère les images
    for lbl in lbls:
        img_gen = gen_carre(img_taille, lbl)
        #mnist.grayramp_show(img_gen)
        #print(60*"=")
        X0.append( np.reshape (img_gen,(nn[0]) ) )
        T.append(resultat_vecteur(lbl))

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
 
















