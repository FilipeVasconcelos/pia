#coding : utf-8

import sys
import numpy as np
from RNmod import reseau 

def quadrant(tab):
    """
    tab: tableau de 2 élements (coordonnées)
    renvoie : une classe de positionnement
                    |
        [0,1,0,0]   |   [1,0,0,0]     
             _______|________
                    |
        [0,0,1,0]   |   [0,0,0,1]
                    |
    """
    if tab[0] > 0 and tab[1] > 0: return [1.,0.,0.,0.]
    if tab[0] < 0 and tab[1] > 0: return [0.,1.,0.,0.]
    if tab[0] < 0 and tab[1] < 0: return [0.,0.,1.,0.]
    if tab[0] > 0 and tab[1] < 0: return [0.,0.,0.,1.]


if __name__ == "__main__" :

    nn  = [2,4,4]

    # ===============
    #  apprentissage
    # ===============
    napp   = 200
    epochs = 50000
    eta    = 0.1 
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1000)

    X0 = 2. * np.random.uniform(size=(napp,2)) - 1.
    T = np.array([quadrant(tab) for tab in X0])
    apprentissage = X0, T
    evaluation=[]
    W,B = RN.gradient_descent( apprentissage, epochs, eta, evaluation )

    # ===============
    #  evaluation
    # ===============
    neval = 1000
    X0 = 20. * np.random.uniform(size=(neval,2)) - 10.
    T  = np.array([quadrant(tab) for tab in X0])
    evaluation = X0,T

    RN= reseau.MCP(nn,verbeux=0,W=W,B=B, verbe_periode=100)
    Y = RN.gradient_descent( apprentissage, 100000, 1.0, evaluation )

    #print(Y)
    cok = 0
    for i,v in enumerate(T):
#        print(i,v) 
        vt = np.argmax(v)
        ve = np.argmax(Y[i])
        coord = X0[i]
        print(coord,ve)
        if vt == ve:
            cok +=1 
    print()
    print()
    print("evaluation score : {:6.2f} %".format(cok/neval*100.))   
























