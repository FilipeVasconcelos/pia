
# coding: utf-8
# XOR basique
import numpy as np
 
iterations = 6000                # Nombre d'itérations

tailleX0, tailleX1, tailleX2 = 2, 3, 1
 
X0 = np.array([[0,0], [0,1], [1,0], [1,1]])
T = np.array([ [0],   [1],   [1],   [0]])
 
def sigmoide (x):
    return 1/(1 + np.exp(-x))    # fonction d'activation
def sigmoide_(x):
    return x * (1 - x)           # dérivée de la fonction d'activation

# Poids
W1 = np.random.uniform(size=(tailleX0, tailleX1))
W2 = np.random.uniform(size=(tailleX1,tailleX2))

for i in range(iterations):
 
    X1 = np.dot(X0, W1)                 # entrée couche 1
    Y1 = sigmoide(X1)                   # activation couche 1
    X2 = np.dot(Y1, W2)                 # entrée couche 2
    Y2 = sigmoide(X2)                   # activation couche 2

    E = T - Y2                          # erreur

    d2 = sigmoide_(Y2) * E              # d2  
    d1 = sigmoide_(Y1) * d2.dot(W2.T)   # d1 

    dW1 = Y1.T.dot(d2)                  # somme sur les entrées des dW1
    dW2 = X0.T.dot(d1)                  # somme sur les entrées des dW2

    W2 += dW1                           # mise à jour des poides de la couche 2
    W1 += dW2                           # et des poids de la couche 1
     
print(Y2)
