#coding: utf-8


if __name__ == "__main__":
    import sys
    import time
    import numpy as np
    import random
    from RNmod import mnist
    from RNmod import reseau

    np.set_printoptions(threshold='nan')
    img_taille = (28,28)

    # =================================
    #              FNN 
    # =================================
    ds_mnist = mnist.lire_ds()


    nn  = [img_taille[0]*img_taille[1],128,64,10] 
        
    X0 = []
    T  = []
    apprentissage = mnist.charger_donnees(dataset="apprentissage")
    print ( "lecture des donnees mnist" ) 

    #napp = len (apprentissage[0])
    napp=60000
    kb= 3000
    nb = int ( napp / kb ) 
    evaluation=[]
    W=None
    B=None
    for kk in range(10):
        for k in range(kb):
            print("batch : ", k,"/",kb)
            X0 = apprentissage[0][k*nb:(k+1)*nb-1]
            T  = apprentissage[1][k*nb:(k+1)*nb-1]
            batch = X0, T 
            RN = reseau.MCP(nn,verbeux=2,verbe_periode=500,distrib_poids="normale",keyg=False,sigma=1.0,mu=0.,W=W,B=B)
            W, B = RN.gradient_descent( batch, 1000, 0.1, evaluation )
  
    print(W)
    print(B)
    neval=10000
    evaluation = mnist.charger_donnees(dataset= "evaluation")
    X0 = evaluation[0][0:neval]
    T  = evaluation[1][0:neval]
    evalu = X0,T
    RN = reseau.MCP(nn,verbeux=2,verbe_periode=1000,distrib_poids="normale",sigma=0.1,mu=0.,W=W,B=B)
    Y = RN.gradient_descent( apprentissage, 10000, 0.1, evalu )
    
    cok = 0
    cal = 0
    T=evalu[1]
    neval=len(T)
    for i,t in enumerate(T):
        vt = np.argmax(t)
        ve = np.argmax(Y[i])
        if vt == ve:
            print("OK",vt,ve)
            cok += 1
        else:
            #mnist.grayramp_show_(evaluation[0][i])
            mnist.ascii_show_([int(x) for x in evaluation[0][i]*255])
            print("WRONG",vt,ve)
            print(''.join('{}: {:8.6f} '.format(*y) for y in enumerate(Y[i])))
            #print(t)
        cal += 1
    print(cal,cok)
    print("evaluation score : {:6.2f} %".format((cok/neval)*100.))
 
