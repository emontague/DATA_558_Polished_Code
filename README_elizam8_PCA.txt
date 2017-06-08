README 

Algorithm: PCA
For course: DATA 558 
By: Libby Montague 
Date: 6/7/2017

Required Python Libraries:
    numpy 
    random
    pandas 
    sklearn
    matplotlib.pyplot 
    copy
    os
    sys

Python scripts:

python elizam8_pca_sample.py [output directory]
    run my PCA algorithm on sample data from a random distribution 
    prints out the number of components needed to get to 95% explained variance ratio
    create a graph comparing the first three principal components 

python elizam8_pca_example.py [output directory]
    run my PCA algorithm on example data from sklearn datasets 'MNIST original'
    the number of classes are subset down and the number of examples are reduced to speed up processing time
    prints out the number of components needed to get to 95% explained variance ratio
    create a graph comparing the first three principal components

python elizam8_pca_sklearn_compare.py [output directory] [comparison type = example or sample]
    run my PCA algorithm on sample or example data 
    sample data are from a random distribution with the mean offset 
    example data is from the sklearn datasets 'MNIST original'
    the number of classes are subset down and the number of examples are reduced to speed up processing time
    prints out the number of components needed to get to 95% explained variance ratio
    create a graph comparing the first three principal components

elizam8_pca.py
    base python script for my pca methods 
    these methods are defined and used in the three demo scripts 

__init__.py 
    empty python script required for importing the elizam8_pca.py methods 