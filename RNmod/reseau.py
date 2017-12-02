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
    def __init__( self, neurones=[2,2,1], mu=0.0, sigma = 1.0,verbeux=0, W=None , B=None , verbe_periode=1 , distrib_poids = "normale" ):

        self.neurones                  = neurones
        self.nombre_de_couches         = len(neurones) - 1 
        self.nombre_de_couches_cachees = self.nombre_de_couches - 1

#        self.dimension_couches_cachees = neurones[1:-1]
#        self.dimension_entree          = neurones[0] 
#        self.dimension_sortie          = neurones[-1] 

        self.sigma_distnormale         = sigma
        self.mu_distnormale            = mu  
        self.distrib_poids             = distrib_poids

        if not (W or B) :

            if self.distrib_poids not in ["normale","uniforme"] : 
                print(distrib_poids,'doit être',["normale","uniforme"])
            if self.distrib_poids == "normale" :
                self.B                 = [np.random.randn(1, y) * sigma + mu for y in neurones[1:]] 
                self.W                 = [np.random.randn(x, y) * sigma + mu for x, y in zip(neurones[:-1], neurones[1:])]
            elif self.distrib_poids == "uniforme" :
                self.B                 = [np.random.uniform(0.,1.,(1, y)) * sigma for y in neurones[1:]] 
                self.W                 = [np.random.uniform(0.,1.,(x, y)) * sigma for x, y in zip(neurones[:-1], neurones[1:])]
        else:
            self.B                     = B
            self.W                     = W
            self.nombre_de_couches     = len(W)
            self.nombre_de_couches_cachees = self.nombre_de_couches - 1

        self.str_sep                   = 20*"====" 
        self.str_sep_                  = 6*"+---" 
        self.verbeux                   = verbeux
        self.verbe_periode             = verbe_periode

    #======================================================================
    def get_tailles(self):
        return len(self.W)

    #======================================================================
    def afficher_poids(self):
        print("Poids")
        print("c n     W")
        print(self.str_sep) 
        np.set_printoptions(precision=6,formatter={'float_kind':float_formatter})
        for i,w in enumerate(self.W):
            for j,wi in enumerate(w) :
                print(i,j,"\n", np.array_str(wi, precision=6, max_line_width=75) )
            print(self.str_sep)
    #======================================================================
    def afficher_biais(self):
        print("Biais")
        print("c n     B")
        print(self.str_sep) 
        for i,b in enumerate(self.B):
            for j,bi in enumerate(b):
                print(i,j,bi)
            print(self.str_sep) 
    #======================================================================
    def preparer_donnees (self, donnees):

        T=np.asarray(donnees[1])
        X0=np.asarray(donnees[0])
        return X0,T

        
    #======================================================================
    def __str__( self ):
        return "MCPerceptron: Struct: {} Dimension: {}".format(len(self.neurones),self.neurones)
    #======================================================================

    def gradient_descent(self, apprentissage, iterations, taux_apprentissage, evaluation=[]):

        X0, T = self.preparer_donnees(apprentissage)
        if len(evaluation) > 0 :
            if self.verbeux > 10 : 
                print( self.str_sep ) 
                print(30*" "+"EVALUATION")
                print( self.str_sep ) 
            X0, T = self.preparer_donnees(evaluation)
            print(X0)
            print(T)
            # ====================
            #     feedforward
            # ====================
            Y = []
            Y.append(X0)
            for k in range( self.nombre_de_couches ) :
                print(k)
                if self.verbeux > 10 : 
                    print(self.str_sep) 
                    print("Y",k,Y[k])
                    print("W",k,self.W[k])
                    print("B",k,self.B[k])
                    print(self.str_sep) 
                X = np.dot(Y[k], self.W[k]) + self.B[k]     # entrée 
                if k == self.nombre_de_couches - 1 :
                    Y.append( X )                     # activation 
                else:
                    Y.append( sigmoide(X) )                     # activation 
           
            return Y[-1]

        if self.verbeux > 10 : 
            print( self.str_sep ) 
            print(30*" "+"APPRENTISSAGE")
            print( self.str_sep ) 
            print()
        print(6*" "+self.str_sep_ )
        print("{:>10s} {:>14s}".format( "pas" , "erreur" ) )
        print(6*" "+self.str_sep_ )
        for pas in range(iterations):

            # ====================
            #     feedforward
            # ====================
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
                if k == self.nombre_de_couches - 1 :
                    Y.append( X )                     # activation 
                else:
                    Y.append( sigmoide(X) )                     # activation 
                if self.verbeux > 10 : 
                    print(k,Y)

            if self.verbeux > 4 : 
                print (Y[-1])

            # ====================
            #        ERREUR 
            # ====================
            E = (T - Y[-1] ) / len (Y)   # erreur
            
            if self.verbeux > 1 and pas%self.verbe_periode==0 : print ( "{:10d} {:18.6e} ".format( pas,(E**2).sum()) )

            # ====================
            #       backprop 
            # ====================
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

                if k == self.nombre_de_couches - 1 :
                    nabla[k]  =  Y[k+1] * nabla[k+1].dot(self.W[k+1].T) 
                else:
                    nabla[k]  =  sigmoide_(Y[k+1]) * nabla[k+1].dot(self.W[k+1].T) 
                dW[k]     =  Y[k].T.dot(nabla[k]) 
                dB[k]     =  np.sum(nabla[k],axis=0)
                self.W[k] += dW[k]
                self.B[k] += dB[k]

            if self.verbeux > 10 :
                print( dW )
                print( nabla )
        print()
        print()
        print(self.str_sep)
        print("convergence après {} pas : {:<18.6e}".format(pas+1,(E**2).sum() ) )
        print(self.str_sep)
        if self.verbeux > 10 :
            print()
            print()
            self.afficher_poids()
            print()
            self.afficher_biais()
            print()
            print()
        if self.verbeux > 1 : 
            print (Y[-1])


        return self.W, self.B 


    #======================================================================

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


















