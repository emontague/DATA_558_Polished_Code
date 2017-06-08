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

## check the number of arguments 
if len(sys.argv) != 3:
    print('format = python elizam8_pca_sample.py [output directory] [comparison type = example or sample]')

else:
    ## read in the arguments 
    outDir = sys.argv[1].strip()
    compare = sys.argv[2].strip()
    print('output to directory = ',outDir)

    error = False

    ## check to see the type of compare
    if compare == 'example':
        ## now read in example data 
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
        ## subset the data randomly to reduce the processing time
        indexes=random.sample(indexes,k=50)
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

    elif compare == 'sample':
        ## now create sample data from normal random distribution
        ## the means are offset to show differentiation in the PCA
        nm.random.seed(0)
        m=50
        n=50
        x_data=nm.concatenate((nm.random.normal(loc=0,scale=1,size=[m,n]),
                            nm.random.normal(loc=20,scale=10,size=[m,n]),
                            nm.random.normal(loc=100,scale=25,size=[m,n]),
                            nm.random.normal(loc=500,scale=100,size=[m,n]),
                            nm.random.normal(loc=1000,scale=200,size=[m,n])),
                            axis=0)
        y_data=nm.concatenate((nm.repeat(0,repeats=m),nm.repeat(1,repeats=m),nm.repeat(2,repeats=m),nm.repeat(3,repeats=m),nm.repeat(4,repeats=m)))

    else:
        ## print an error if the input is incorrect 
        error = True
        print('error with comparison input must be either example or sample')

    if error == False:

        ## standardize the data 
        ## for pca the mean = 0 
        mean_data=nm.mean(x_data,axis=0)
        data=x_data-mean_data

        print('standardized data')

        ## run my pca
        my_pca=elizam8_pca.pca(d=data,n=data.shape[1],m=100)

        print('my pca complete')

        ## run sklearn pca
        sklearn_pca=dc.PCA(n_components=25)
        print('sklearn pca complete')
        sklearn_pca.fit(data)

        ## check the first component
        print("my first component = ")
        print(my_pca[0,])
        ## compare with sklearn 
        print("sklearn 1st Principal Component = ")
        print(sklearn_pca.components_[0,])
        ## check the second component
        print("my second component = ")
        print(my_pca[1,])
        ## compare with sklearn 
        print("sklearn 2nd Principal Component = ")
        print(sklearn_pca.components_[1,])

        ## check the explained variance   
        print('compare explained variance')
        print('my pca')
        elizam8_pca.test_95(elizam8_pca.calcEVR(p=my_pca,x=data))
        print('sklearn pca')
        elizam8_pca.test_95(sklearn_pca.explained_variance_ratio_)

        ## create plots to compare the pca methods
        my_pca_transform=x_data.dot(my_pca.T)
        sklearn_transform=sklearn_pca.transform(data)
        elizam8_pca.drawCompare(d1=my_pca_transform,l1=y_data,n1='My',d2=sklearn_transform,l2=y_data,n2='sklearn',f=outDir+'/elizam8_pca_compare_'+compare+'_data.jpeg')
        print('created graph to compare pca = ',outDir+'/elizam8_pca_compare_'+compare+'_data.jpeg')