#coding: utf-8

import sys
import random
import numpy as np
import inspect

from .activation import *
#import activation
#import sigmoide, sigmoide_

float_formatter = lambda x: "%12.8f" % x

class MCP():

    """
    ======================================================================
    Multi-Couches Perceptron:
    cette classe est un exemple d'implémentation d'un réseau de neurones 
    "feedforward" à apprentissage profond.
    Ce code est destiné à l'enseignement de la mineure :
    "Initiation à l'intelligence artificiel" 
    pour les INGE1 de l'ESME Sudria Lille.
    date 31/10/17
    ======================================================================
    auteurs : - G. Roux        (enseignant des mathématiques/informatique)
              - F. Vasconcelos (enseignant de SI/informatique)
    ======================================================================
    """

    #======================================================================
    def __init__( self, neurones=[2,2,1], mu=0.0, sigma = 1.0,verbeux=0, W=None , B=None ):

        self.neurones                  = neurones
        self.nombre_de_couches         = len(neurones) - 1 
        self.nombre_de_couches_cachees = self.nombre_de_couches - 1

#        self.dimension_couches_cachees = neurones[1:-1]
#        self.dimension_entree          = neurones[0] 
#        self.dimension_sortie          = neurones[-1] 

        self.sigma_distnormale         = sigma
        self.mu_distnormale            = mu  

        if not (W or B) :
            self.B                     = [np.random.randn(1, y) * sigma + mu for y in neurones[1:]] 
            self.W                     = [np.random.randn(x, y) * sigma + mu for x, y in zip(neurones[:-1], neurones[1:])]
        else:
            self.B                     = B
            self.W                     = W
            self.nombre_de_couches_cachees = len(W)
            self.nombre_de_couches = self.nombre_de_couches_cachees + 1 

        self.str_sep                   = 20*"+---" 
        self.verbeux                   = verbeux
        self.print_rate                = 1000

    #======================================================================
    def get_tailles(self):
        return len(self.W)

    #======================================================================
    def afficher_poids(self):
        print("poids")
        print("c n     w")
        print(self.str_sep) 
        np.set_printoptions(precision=6,formatter={'float_kind':float_formatter})
        for i,w in enumerate(self.W):
            for j,wi in enumerate(w) :
                print(i,j,"\n", np.array_str(wi, precision=6, max_line_width=75) )
            print(self.str_sep) 
    #======================================================================
    def preparer_donnees (self, donnees):

#        self.W[0][:,-1]=0.
#        self.W[0][-1,-1]=1.
#        self.W[1][:,-1]=0.
#        self.W[1][-1,-1]=1.
#        print(self.W[0],self.W[0][0][-1])
#        print(self.W[1])
        #sys.exit()

        T=np.asarray(donnees[1])

#        app_save=donnees[0]
#        X=[]
#        for a in donnees[0] :
#            a.append(1.0) 
#            X.append(a)
#        X0=np.asarray(X)

        X0=np.asarray(donnees[0])
        return X0,T

        
    #======================================================================
    def __str__( self ):
        return "MCPerceptron: Struct: {} Dimension: {}".format(len(self.neurones),self.neurones)
    #======================================================================

    def gradient_descent(self, apprentissage, iterations, taux_apprentissage, evaluation=[]):

        X0, T = self.preparer_donnees(apprentissage)
        #print(X0)
        #print(T)
        if len(evaluation) > 0 :
            X0, T = self.preparer_donnees(evaluation)
            print(X0)
            print(T)
            #feedforward
            Y = []
            Y.append(X0)
            for k in range( self.nombre_de_couches ) :
                X = np.dot(Y[k], self.W[k]) + self.B[k]     # entrée 
                Y.append( sigmoide(X) )                     # activation 
           
            print( Y[-1] )

        #print(X0)
        #print(T)
        #sys.exit()
        for pas in range(iterations):

            #feedforward
            Y = []
            Y.append(X0)
            if self.verbeux > 10 : 
                print(self.str_sep) 
                print ("feedforward")
                print(self.str_sep) 
            for k in range( self.nombre_de_couches ) :
                if self.verbeux > 10 : 
                    print(self.str_sep) 
                    print("Y",k,Y[k])
                    print("W",k,self.W[k])
                    print("B",k,self.B[k])
                    print(self.str_sep) 
                X = np.dot(Y[k], self.W[k]) + self.B[k]     # entrée 
                Y.append( sigmoide(X) )                     # activation 
                if self.verbeux > 10 : 
                    print(k,Y)

            if self.verbeux > 10 : 
                print (Y[-1])

            # ====================
            #        ERREUR 
            # ====================
            E = T - Y[-1]                                    # erreur
            
            if self.verbeux > 1 and pas%self.print_rate==0 : print (pas,(E**2).sum() )

            nabla = len(self.W) * [None] 
            dW    = len(self.W) * [None] 
            dB    = len(self.B) * [None] 
            nabla[-1]   =  taux_apprentissage * sigmoide_(Y[-1]) * E              
            dW[-1]      = Y[-2].T.dot(nabla[-1])
            dB[-1]      = np.sum(nabla[-1],axis=0)
            self.W[-1] += dW[-1] 
            self.B[-1] += dB[-1] 
            if self.verbeux > 10 :
                print(nabla[-1])
                print(nabla[1])
                print(sigmoide_(Y[0]))

            for k in reversed(range(self.nombre_de_couches_cachees)) :
                if self.verbeux > 10 :
                    print("nabla delta loop",k)
                    print("sigmoide' Y[k+1]",sigmoide_(Y[k+1]))
                    print("nabla k+1", nabla[k+1] )
                    print("W[k+1]",self.W[k+1] )

                nabla[k]  =  sigmoide_(Y[k+1]) * nabla[k+1].dot(self.W[k+1].T) 
                dW[k]     =  Y[k].T.dot(nabla[k]) 
                dB[k]     =  np.sum(nabla[k],axis=0)
                self.W[k] += dW[k]
                self.B[k] += dB[k]

            if self.verbeux > 10 :
                print( dW )
                print( nabla )
        print (pas,(E**2).sum() )
        #print(Y[-1] )
        #print(self.W)
        #print(self.B)
        return self.W, self.B 
#
#    d3 = sigmoide_(Y3) * E              # d2  
#    dW3 = Y2.T.dot(d3)                  # somme sur les entrées des dW1
#    W3 += dW3                           # mise à jour des poides de la couche 2

#    d2 = sigmoide_(Y2) * d3.dot(W3.T)   # d1 
#    dW2 = Y1.T.dot(d2)                  # somme sur les entrées des dW2
#    W2 += dW2                           # et des poids de la couche 1
#
#    d1 = sigmoide_(Y1) * d2.dot(W2.T)   # d1 
#    dW1 = Y0.T.dot(d1)                  # somme sur les entrées des dW2
#    W1 += dW1                           # et des poids de la couche 1




if __name__ == "__main__" :

    nn  = [2,5,1]


    RN = MCP(nn)
    
    X0 = [ [ 0., 0. ] , 
           [ 0., 1. ] , 
           [ 1., 0. ] , 
           [ 1., 1. ] ] 

    T  =   [ [ 0. ] ,
             [ 1. ] ,
             [ 1. ] ,
             [ 0. ] ] 
    apprentissage = X0, T

    X0 = [ [ 0., 0. ] , 
           [ 1., 0. ] ] 

    T  = [ [ 0. ] ,
           [ 1. ] ] 
    evaluation = X0, T
    
    if False:
        for iseed in range(59,10000) :
            print(iseed)
            np.random.seed(iseed)
            RN = MCP(nn)
            RN.gradient_descent( apprentissage, 30000, 1., evaluation )

    RN = MCP(nn,verbeux=2)
    RN.gradient_descent( apprentissage, 100000, 10.0, evaluation )


















