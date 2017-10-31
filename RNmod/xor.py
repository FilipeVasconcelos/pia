#-*- coding: utf-8 -*-

import sys
import numpy as np

class Reseau():

    def __init__( self, nombre_de_couches_cachees, dimension_entree, dimension_sortie ):

        self.nombre_de_couches_cachees = nombre_de_couches_cachees
        self.dimension_entree          = dimension_entree
        self.dimension_sortie          = dimension_sortie

# ===========================================================================
def xor_dur(entree1,entree2):
    
    yi = np.array( [ 0., 0.] )
    if entree1: yi[0]=1.
    if entree2: yi[1]=1.
    #print( yi )
    wij = np.array( [ [  1., 1.], 
                      [ -1., -1.] ] )
    bi  = np.array( [ 0.5, -1.5 ] ) 
    oi  = np.array( [ 1.0,  1.0 ] ) 
    bz  = 1.5
    ai  = wij.dot(yi) 
    #print( ai )
    z = 0.
    for a,b,o in zip(ai,bi,oi):
        if a > b:
            z += o
    if z > bz :
        return True 
    else:
        return False

# ===========================================================================
def xor_mou(entree1,entree2):

    nombre_de_couches = 2
    yi  = np.zeros( (1,2) , dtype=float )
    if entree1: yi[0,0]=1.
    if entree2: yi[0,1]=1.

    key =  "s"
#    key =  "h"
                           
    #wij = np.array( [[  [  1., -1.]  ,  #w^(1)_{11} w^(1)_{12}
    #                    [ -1.,  1.] ],  #w^(1)_{21} w^(1)_{22}
    #                  [ [  1.,  1.]  ,  #w^(2)_{11} w^(2)_{12}
     #                   [  0.,  0.] ]] )#w^(2)_{21} w^(2)_{22}      
     
    wij = np.array( [ [[ 2.12096563,  6.46646883],
                       [ 2.12093506,  6.46457309]],
                      [[ -1.04650633e+01,   6.39975485e-20],
                       [  1.03550243e+01,  -5.96282777e-20]] ] )

    if key == "s" :
        bi  = np.array(    [[  0. ,  0.]  ,
                            [  0. ,  0.]]   ) 
    elif key == "h":
        bi  = np.array(    [[  0.,  0.]  ,
                            [  0.,  0.]]   ) 
    elif key == "t":
        bi  = np.array(    [[  0.,  0.]  ,
                            [ -0.76159416, 0.76159416]]   ) 
#    print( bi[0,:] )
    #bi  = np.zeros((2,2),dtype=float)

    verbeux = 1

    ai  = np.zeros((nombre_de_couches,2),dtype=float)
#    print(ai[:,0],yi[0,:])     
    for k in range( nombre_de_couches ) :
        if key  == "s" :
            ai[k,:]  = sigmoide  ( np.dot(yi[0,:],wij[k,:,:])  - bi[k,:] )
        elif key == "h":
            ai[k,:]  = activation_heaviside ( wij[k,:,:].dot(yi[0,:])  - bi[k,:] )
        elif key == "t":
            ai[k,:]  = activation_tanh      ( wij[k,:,:].dot(yi[0,:])  - bi[k,:] )
        if verbeux > 10 :
            print(30*"=")
            print("k =",k)
            print(30*"=")
            print("yi", yi[0,:] )
            print()
            print("bi", bi[k,:] )
            print()
            print("wij", wij[k,:,:]  )
            print()
            print("wij . yi"              , wij[k,:,:].dot(yi[0,:])                     )
            print("wij . yi - bi"         , wij[k,:,:].dot(yi[0,:]) - bi[k,:]           )
            if key  == "s" :
                print("phi ( wij . yi - bi ) ", sigmoide ( wij[k,:,:].dot(yi[0,:]) - bi[k,:] ) )
            elif key  == "h" :
                print("phi ( wij . yi - bi ) ", activation_heaviside ( wij[k,:,:].dot(yi[0,:]) - bi[k,:] ) )
            elif key  == "t" :
                print("phi ( wij . yi - bi ) ", activation_tanh ( wij[k,:,:].dot(yi[0,:]) - bi[k,:] ) )
            print()
            print("ai",ai[k,:])
            print()
        yi[0,:]  = ai[k,:]
        

    return yi[0,0]

# ===========================================================================
def activation_heaviside(vecteur) :

    for i,e in enumerate(vecteur):
        if e > 0. :
            vecteur[i] = 1. 
        else:
            vecteur[i] = 0.

    return vecteur
# ===========================================================================
def activation_sigmoide(vecteur):

    for i,e in enumerate(vecteur):
        vecteur[i] = 1./(1+np.exp(-e))

    return vecteur

def sigmoide(x):
    return 1./(1+np.exp(-x))

# ===========================================================================
def activation_tanh(vecteur):

    for i,e in enumerate(vecteur):
        vecteur[i] = np.tanh(e)

    return vecteur


if __name__ == "__main__" :

    print(60*"=")
    print("voila le super héro XOR !!!")
    print(60*"=")

    #ncc = 2
    #de  = 5
    #ds  = 3

    entrees  = [ ( False, False ) , 
                 ( False, True  ) , 
                 ( True , False ) , 
                 ( True , True  ) ]
    for entree in entrees :
        print ( "mou",entree, xor_mou( entree[0], entree[1] ) )
        print ( "dur",entree, xor_dur( entree[0], entree[1] ) )
#    print ( xor_dur( True, False )  )
#    print ( xor_mou( True, False )  ) 
























