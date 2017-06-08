## import the libraries
import numpy as nm
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import copy
import sklearn.decomposition as dc
import sklearn.preprocessing as pre
import sklearn.datasets as ds
import random
import os
import sys

## import my python file 
import elizam8_pca

## check the number of system arguments 
if len(sys.argv) != 2:
    print('format = python elizam8_pca_sample.py [output directory]')

else:
    ## store the output directory 
    outDir = sys.argv[1].strip()
    print('output to directory = ',outDir)

    ## read in the data 
    mnist=ds.fetch_mldata('MNIST original')
    ## subset classes to reduce the processing time 
    print('shape of original y data = ',mnist.target.shape)
    print('shape of original x data = ',mnist.data.shape)
    indexes=list()
    i=0
    for i in range(0,len(mnist.target)):
        y=mnist.target[i]
        if y in [1,2,3]: indexes.append(i)
    ## subset the data randomly in order to reduce the processing time 
    indexes=random.sample(indexes,k=100)
    ## create the final data
    y_data=list()
    x_data=nm.zeros((len(indexes),mnist.data.shape[1]))
    n=0
    for i in indexes:
        x_data[n,:]=mnist.data[i,:]
        y_data.append(mnist.target[i])
        n+=1
    y_data=nm.array(y_data)
    print('shape of y data = ',y_data.shape)
    print('shape of x data = ',x_data.shape)
    print('classes in sample data = ',nm.unique(y_data))

    ## standardize the data 
    ## for pca the input should have mean = 0 
    mean_data=nm.mean(x_data,axis=0)
    data=x_data-mean_data

    ## run my pca algorithm 
    ## returns all possible principal components 
    my_pca=elizam8_pca.pca(d=data,n=data.shape[1],m=100)

    ## calculate the number of principal components to reach 95% 
    elizam8_pca.test_95(elizam8_pca.calcEVR(p=my_pca,x=data))

    ## create a graph comparing the 1/2 and 2/3 components 
    print('creating graph of data')
    elizam8_pca.drawGraphs(d_result=x_data.dot(my_pca.T),labels=y_data,f=outDir+'/elizam8_pca_example_data.jpeg')    