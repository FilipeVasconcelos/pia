
# coding: utf-8
import numpy as np
 
iterations = 40000           # Nombre d'itérations
tailleX0, tailleX1, tailleX2 = 2, 4, 4

def quadrant(tab):
    if tab[0] > 0 and tab[1] > 0: return [1,0,0,0]
    if tab[0] < 0 and tab[1] > 0: return [0,1,0,0]
    if tab[0] < 0 and tab[1] < 0: return [0,0,1,0]
    if tab[0] > 0 and tab[1] < 0: return [0,0,0,1]

def max_a_un(tab):
    m = np.max(tab)
    return np.array([0 if e < m else 1 for e in tab])

X0 = 20 * np.random.uniform(size=(20,2)) - 10
T = np.array([quadrant(tab) for tab in X0])
 
def sigmoide (x): return 1/(1 + np.exp(-x))  # fonction d'activation
def sigmoide_(x): return x * (1 - x)         # dérivée de la fonction d'activation

# Poids
W1 = np.random.uniform(size=(tailleX0, tailleX1))
W2 = np.random.uniform(size=(tailleX1,tailleX2))

eta = 1

for i in range(iterations):
 
    X1 = np.dot(X0, W1)                 # entrée couche 1
    Y1 = sigmoide(X1)                   # activation couche 1
    X2 = np.dot(Y1, W2)                 # entrée couche 2
    Y2 = sigmoide(X2)                   # activation couche 2

    E = T - Y2                          # erreur

    d2 = sigmoide_(Y2) * E              # d2  
    d1 = sigmoide_(Y1) * d2.dot(W2.T)   # d1 

    W2 +=  Y1.T.dot(d2)                 # mise à jour des poides de la couche 2
    W1 +=  X0.T.dot(d1)                 # et des poids de la couche 1

print(W1)
print(W2)

TTE = 20 * np.random.uniform(size=(1000,2)) - 10
TT = np.array([quadrant(tab) for tab in TTE])
print(TTE)
X1 = np.dot(TTE, W1)                # entrée couche 1
Y1 = sigmoide(X1)                   # activation couche 1
X2 = np.dot(Y1, W2)                 # entrée couche 2
Y2 = sigmoide(X2)                   # activation couche 2

#Y22 = np.array([[1 if (v > 0.6) else 0 for v in tab] for tab in Y2])
#Y22 = np.array([max_a_un(tab) for tab in Y2])
Y22 = np.array([np.argmax(tab) for tab in Y2])
TTT = np.array([np.argmax(tab) for tab in TT])
print(Y22)
print(TTT)
print(np.sum(Y22 == TTT) / 1000.)









