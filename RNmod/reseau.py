#-*- coding: utf-8 -*-

import sys
import random
import numpy as np
import inspect

import activation

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
    def __init__( self, neurones=[2,2,1], mu=0.0, sigma = 1.0):

        self.neurones                  = neurones
        self.nombre_de_couches_cachees = len(neurones) - 1
        self.dimension_couches_cachees = neurones[1:-1]
        self.dimension_entree          = neurones[0] 
        self.dimension_sortie          = neurones[-1] 
        self.sigma_distnormale         = sigma
        self.mu_distnormale            = mu  
        self.biais                     = [np.random.randn(y, 1) * sigma + mu for y in neurones[1:]]
        self.poids                     = [np.random.randn(y, x) * sigma + mu for x, y in zip(neurones[:-1], neurones[1:])]
        self.str_sep                   = 20*"+---" 

    #======================================================================
    def get_tailles(self):
        return len(self.poids), len(self.biais)

    #======================================================================
    def afficher_poids(self):
        print("poids")
        print("c n     w")
        print(self.str_sep) 
        np.set_printoptions(precision=6,formatter={'float_kind':float_formatter})
        for i,w in enumerate(self.poids):
            for j,wi in enumerate(w) :
                print(i,j,"\n", np.array_str(wi, precision=6, max_line_width=75) )
            print(self.str_sep) 
    #======================================================================
    def afficher_biais(self):
        print("biais")
        print("c n     b")
        print(self.str_sep) 
        for i,b in enumerate(self.biais):
            for j,bi in enumerate(b):
                print(i,j,bi)
            print(self.str_sep) 
    #======================================================================
    def __str__( self ):
        return "MCPerceptron: Struct: {} Dimension: {}".format(len(self.neurones),self.neurones)
    #======================================================================
    def gradient_descent(self, apprentissage, iterations, taux_apprentissage, evaluation=[]):

        for pas in range(iterations):
            random.shuffle( apprentissage )
            

    #======================================================================

if __name__ == "__main__" :

    nn  = [2,3,1]

    RN = MCP(nn)
    #print( inspect.getdoc(RN) )
    print(RN)

    print ( RN.get_tailles() )
    RN.afficher_poids() 
    RN.afficher_biais() 

    entree = [ [ 0., 0. ] , 
               [ 0., 1. ] , 
               [ 1., 0. ] , 
               [ 1., 1. ] ]

    sortie = [ [ 0. ] ,
               [ 1. ] ,
               [ 1. ] ,
               [ 0. ] ]
    apprentissage = (entree,sortie)
   
    
    entree = [ [ 0., 0. ] , 
               [ 1., 0. ] ]

    sortie = [ [ 0. ] ,
               [ 1. ] ]
    evaluation = entree,sortie

    print( apprentissage[0] ) 

    RN.gradient_descent( apprentissage, 6000, 1., evaluation )



















