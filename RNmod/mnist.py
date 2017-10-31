# coding : utf-8
import os
import sys
import struct
import time
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
from term_colors import *

"""
inspiré de http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
qui est sous licence GPL licensed.
"""

def lire_ds (dataset= "apprentissage", path = "."):
    """
    fonction Python pour importer les données MNIST.
    Elle renvoie un itérateur d'un tuple de dimension 2 :
     - 1er element le label de l'image
     - numpy.uint8 tableau 2D des pixels (28x28)
    """

    if dataset is "apprentissage":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "evaluation":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset doit prendre la valeur 'apprentissage' or 'evaluation'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def charger_donnees(dataset= "apprentissage", path = "." ):

    mnist = lire_ds( dataset, path )
    entree = []
    sortie = []
    while True :
        try:
            image = next(mnist)
        except StopIteration:
            break  # Iterator exhausted: stop the loop
        else:
            entree.append( np.reshape       ( image[1], (784) ) ) 
            sortie.append( resultat_vecteur ( image[0])           )  
    return entree, sortie

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def ascii_show(image):
    """
    Ascii render 2D
    """
    for y in image:
        row = ""    
        for x in y:
            row += '{0: <3}'.format(x)
        print(row)

def ascii_show_(image):
    """
    Ascii render 1D 
    """
    row = ""    
    for i,x in enumerate(image):
        if i%28 == 0 :
            print(row)
            row = ""    
        row += '{0: <3}'.format(x)

def grayramp_show(image):
    """
    gray render 1D 
    """
    for y in image:
        row = ""
        for x in y:
            print_color('  ', bg=gray(x/23), end='')

        print(row)

def grayramp_show_(image):
    """
    gray render 1D 
    """
    row = ""    
    for i,x in enumerate(image):
        print_color('  ', bg=gray( x/24 ), end='')
        if i%28 == 0 :
            print(row)


def resultat_vecteur(e):
    """
    Transforme le label en un vecteur de 10 elements
    """
    v = np.zeros((10))
    v[e] = 1.0
    return v
                         
if __name__ == "__main__" :

    mnist = lire_ds()

    i=0
    while i < 1 :
        image = next(mnist)
        ascii_show(image[1])
        grayramp_show ( image[1] )
        #show(image[1])
        i+=1

    evaluation_img, evaluation_lbl = charger_donnees()
    print( len  ( evaluation_img[0] ) , len ( evaluation_lbl[0] )) 

    k = 254
    ascii_show_ ( evaluation_img[k] )
    print( evaluation_lbl[k] )

    print()
    grayramp_show_ ( evaluation_img[k] )
    print()
    print()
    print()

    for img in evaluation_img:
        grayramp_show_ (img) 
        time.sleep(0.1)        


