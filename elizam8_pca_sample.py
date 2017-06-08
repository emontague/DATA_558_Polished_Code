## import the libraries
import numpy as nm
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import copy
import sklearn.decomposition as dc
import sklearn.preprocessing as pre
import os
import sys

## import my python file 
import elizam8_pca

## check the number of arguments 
if len(sys.argv) != 2:
    print('format = python elizam8_pca_sample.py [output directory]')

else:
    ## store the arguments
    outDir = sys.argv[1].strip()
    print('output to directory = ',outDir)

    ## generate the random data
    ## the means are offset in order to show differentiation using the PCA 
    nm.random.seed(0)
    m=50
    n=50
    org_data=nm.concatenate((nm.random.normal(loc=0,scale=1,size=[m,n]),
                        nm.random.normal(loc=20,scale=10,size=[m,n]),
                        nm.random.normal(loc=100,scale=25,size=[m,n]),
                        nm.random.normal(loc=500,scale=100,size=[m,n]),
                        nm.random.normal(loc=1000,scale=200,size=[m,n])),
                        axis=0)
    classes=nm.concatenate((nm.repeat(0,repeats=m),nm.repeat(1,repeats=m),nm.repeat(2,repeats=m),nm.repeat(3,repeats=m),nm.repeat(4,repeats=m)))
    
    ## standardize the data 
    ## for pca the data should have mean = 0 
    mean_data=nm.mean(org_data,axis=0)
    data=org_data-mean_data

    ## run my pca algorithm
    ## get back all possible components
    my_pca=elizam8_pca.pca(d=data,n=data.shape[1],m=100)

    ## calcualte where the number of components to reach 95% variance
    elizam8_pca.test_95(elizam8_pca.calcEVR(p=my_pca,x=data))

    ## create a graph comparing the 1/2 and 2/3 components
    print('creating graph of data')
    elizam8_pca.drawGraphs(d_result=org_data.dot(my_pca.T),labels=classes,f=outDir+'/elizam8_pca_sample_data.jpeg')    