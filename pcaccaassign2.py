# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:52:34 2018

@author: saketm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import random
import scipy as sp

#plotting function and correlation
def plotter(x,y):
    for i in range(0,len(x[0])):
        r,p=sp.stats.pearsonr(x[:,i],y[:,i]) 
        plt.scatter(x[:,i], y[:,i])
        plt.show()
        print("Pearson correlation value for column ",i+1 ," of X and Y is", r)

#PCA fitting
def pcafitting(X,Y):
    pca=PCA(n_components=4, copy=True, whiten =False)
    X=pca.fit_transform(X)
    Y=pca.fit_transform(Y)
    print("PCA Correlations")
    plotter(X,Y)

#CCA fitting
def ccafitting(X,Y):
    cca=CCA(n_components=4, copy=True, scale= True, tol= 1e-6)
    Xcca,Ycca =cca.fit_transform(X,Y)
    print("CCA Correlations")
    plotter(Xcca,Ycca)
    
#data generation and function calls
X,Y = np.empty(100).reshape(1,100), np.empty(100).reshape(1,100)
for i in range(0,1000):
    x= np.random.normal(loc=random.randint(1,4), scale=0.05, size =100).reshape(1,100)
    y= np.random.poisson(lam=(random.randint(1,4)*2), size =100).reshape(1,100)
    X=np.append(X,x, axis=0)
    Y=np.append(Y,y, axis=0)
X=X[1:1001,:]
Y=Y[1:1001,:]
#data normalizer
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X=sc.fit_transform(X)
Y=sc.fit_transform(Y)
pcafitting(X,Y)
ccafitting(X,Y)

