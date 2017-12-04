#coding: utf-8

import sys
import numpy as np
import random
from RNmod import mnist
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

def gen_triangle(taille,tkey): 

    img = np.zeros(taille,dtype=int) 
    xo =np.random.randint(tkey,28-tkey)
    yo =np.random.randint(tkey,28-tkey)
    print(" centre du triangle ",xo,yo)
    val = 1.
    if tkey == 3:
        img[xo,yo ] = val
        img[xo+1,yo-1:yo+2 ] = val
    if tkey == 5:
        img[xo,yo ] = val
        img[xo+1,yo-1:yo+2 ] = val
        img[xo+2,yo-2:yo+3 ] = val
    if tkey == 7:
        img[xo,yo ] = val
        img[xo+1,yo-1:yo+2 ] = val
        img[xo+2,yo-2:yo+3 ] = val
        img[xo+3,yo-3:yo+4 ] = val


    return img


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
    img_gen = gen_carre(img_taille, 3)
    mnist.grayramp_show(img_gen)
    print(60*"=")
    img_gen = gen_carre(img_taille, 5)
    mnist.grayramp_show(img_gen)
    print(60*"=")
    img_gen = gen_carre(img_taille, 7)
    mnist.grayramp_show(img_gen)
    print(60*"=")

    img_gen = gen_triangle(img_taille, 3)
    mnist.grayramp_show(img_gen)
    img_gen = gen_triangle(img_taille, 5)
    mnist.grayramp_show(img_gen)
    img_gen = gen_triangle(img_taille, 7)
    mnist.grayramp_show(img_gen)


