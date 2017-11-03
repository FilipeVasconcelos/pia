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

    RN = reseau.MCP(nn,verbeux=2,verbe_periode=10000)

    X0 = 20. * np.random.uniform(size=(20,2)) - 10.
    T = np.array([quadrant(tab) for tab in X0])
    apprentissage = X0, T

    evaluation=[]
    W,B = RN.gradient_descent( apprentissage, 50000, 2.0, evaluation )

    X0 = 20. * np.random.uniform(size=(1000,2)) - 10.
    T  = np.array([quadrant(tab) for tab in X0])
    evaluation = X0,T

    RN= reseau.MCP(nn,verbeux=11,W=W,B=B, verbe_periode=100)
    RN.gradient_descent( apprentissage, 100000, 1.0, evaluation )









