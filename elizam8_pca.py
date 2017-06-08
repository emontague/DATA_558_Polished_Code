## import the libraries
import numpy as nm
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import copy
import sklearn.decomposition as dc
import sklearn.preprocessing as pre

## modified from my homework 5
## define the oja algorithm
## inputs: 
## d = data
## m = max
## a = starting a
## t_0 = starting stepsize 
## output: 
## a = principal component 
## result = top eignvalue for each of the eigenvectors 
## finalTen = average of the first top 10 a 
def oja(d,m,a_0,t_0,n_0):
    n=d.shape[0]
    ## go through each iteration 
    i=0
    #print('m = ',m)
    result=nm.zeros(m)
    a=a_0
    t=0
    ## store the last 10 iterations
    finalTen=nm.array(nm.zeros((10,d.shape[1])))
    while i < m:
        ## shuffle the data at each iteration 
        nm.random.shuffle(d)
        #print(d[0,0])
        ## go through each of the n 
        for r in range(0,n):
            ## calculate a_t+1
            nt=n_0/(t+t_0)
            a_t=a+(nt*d[r,:].dot(d[r,:].T.dot(a)))
            ## replace the a with the updated a_t+1
            a=a_t/nm.linalg.norm(a_t)
            t += 1
        #print(a)
        ## now compute the eigenvalue for this top eigenvalue
        ## calculate the top eigenvector according to the equation to solve for the top eigenvalue 
        ## where a = (1/n)*zt*z
        result[i]=a.dot(d.T.dot(d).dot(a))*(1/n)
        ## store the last 10 iterations 
        if i > m-10:
            finalTen[m-i,:]=a
        i += 1
    return a, result, nm.average(finalTen,axis=0)

## function to deflate the data
## inputs:
## a = top eigenvector
## d = data to deflate
## output: deflated data ready to be used to calculate the next principal component  
def deflate(d,a):
    #r0=a.dot(a.T)
    #print(r0.shape)
    ## use this function instead of the one above for a*a^T
    r0=nm.outer(a,a)
    r1=d.dot(r0)
    r2=d-r1
    return(r2)

## run my PCA algorithm 
## inputs:
## d = data
## n = number of principal components 
## m = max iterations 
def pca(d,n,m):
    beta=nm.array(nm.zeros(d.shape[1]))
    a_0=nm.random.normal(size=d.shape[1])
    a_0=a_0/nm.linalg.norm(a_0)
    components=nm.zeros((n,d.shape[1]))
    for p in range(0,n):
        a, maximum, r = oja(d=copy.deepcopy(d),m=m,a_0=copy.deepcopy(a_0),t_0=1,n_0=0.0001)
        components[p,:]=a
        d=deflate(d=copy.deepcopy(d),a=copy.deepcopy(a))
    return components

## calcualte the explained variance ratio
## input 
## p = principal components 
## x = data
## output:
## e = array of the explained variance ratio for each principal component 
def calcEVR(p,x):
    e=nm.zeros(p.shape[0])
    for i in range(0,p.shape[0]):
        e0=nm.sum(x.dot(p[i,])**2)
        e1=nm.sum(nm.sum(x**2,axis=0))
        e[i]=e0/e1
    return(e)

## calculate the number of components needed to reach 95% explained variance ratio
## input
## evr = array of the explained variance ratio for each principal component
## output 
## var_explained = the sum of the explained variance ratios
## prints to the consule the number of components and the reached variance 
def test_95(evr):
    evr_sk_sum=0
    i=1
    hit_95=True
    var_explained=0
    for e in evr:
        evr_sk_sum=evr_sk_sum+e
        if (evr_sk_sum >= 0.95) & hit_95:
            print('Number of Principal Components to get to ',nm.round(evr_sk_sum,decimals=3)*100,'% Variance Explained = ',i)
            var_explained=i
            hit_95=False
        i += 1
    if hit_95 == True: print('Never hit 95% explained variance, total exaplined variance = ',nm.round(evr_sk_sum,decimals=3)*100) 
    return var_explained

## draw a graph to compare the 1/2 and 2/3 components 
## color the classes by different colors 
## prints out the location of the printed graph 
## input
## d_result = transformed data
## labels = y labels 
## f = folder path of the results 
def drawGraphs(d_result,labels,f):
    #print(len(nm.unique(labels)))
    colors=["b","g","r","c","m","y","k"]
    classes=nm.unique(labels)
    #class_map=pd.DataFrame({'classes':nm.unique(labels),'numbers':[0,1,2,3,4]})
    if len(classes) > len(colors): "too many classes for the number of colors, colors will be repeated"
    plt.figure(1)
    plt.subplot(211)
    i=0
    for l in labels:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        #print(c_i,' --> ',int((c_i/len(classes) % 1)*len(classes)),' --> ',c)
        plt.plot(d_result[i,0],d_result[i,1],marker="o",color=c,linestyle="None")  
        i+=1  
    plt.xlabel('1st Component')  
    plt.ylabel('2nd Component')  
    plt.title('PCA Components')
    plt.subplot(212)
    i=0
    for l in labels:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        plt.plot(d_result[i,1],d_result[i,2],marker="o",color=c,linestyle="None")
        i+=1
    plt.xlabel('2nd Component')  
    plt.ylabel('3rd Component')   
    #print(f)
    plt.savefig(f)
 
## draw graphs to compare the first 3 principal components of 2 methods (1, 2)
## inputs:
## d1, d2 = transformed data from method 1 and 2 
## l1, l2 = labels for method 1 and 2 
## n1, n2 = name of the two methods 
## f = folder path for the graph results
def drawCompare(d1,l1,n1,d2,l2,n2,f):
    #print(len(nm.unique(labels)))
    colors=["b","g","r","c","m","y","k"]
    ## start by plotting the first method
    classes=nm.unique(l1)
    #class_map=pd.DataFrame({'classes':nm.unique(labels),'numbers':[0,1,2,3,4]})
    if len(classes) > len(colors): "too many classes for the number of colors, colors will be repeated"
    plt.figure(1)
    plt.subplot(221)
    i=0
    for l in l1:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        #print(c_i,' --> ',int((c_i/len(classes) % 1)*len(classes)),' --> ',c)
        plt.plot(d1[i,0],d1[i,1],marker="o",color=c,linestyle="None")  
        i+=1  
    plt.xlabel('1st Component')  
    plt.ylabel('2nd Component')  
    plt.title(n1+' PCA Components')
    plt.subplot(222)
    i=0
    for l in l1:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        plt.plot(d1[i,1],d1[i,2],marker="o",color=c,linestyle="None")
        i+=1
    plt.xlabel('2nd Component')  
    plt.ylabel('3rd Component')   
    ## now plot the next method 
    classes=nm.unique(l2)
    #class_map=pd.DataFrame({'classes':nm.unique(labels),'numbers':[0,1,2,3,4]})
    if len(classes) > len(colors): "too many classes for the number of colors, colors will be repeated"
    plt.subplot(223)
    i=0
    for l in l2:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        #print(c_i,' --> ',int((c_i/len(classes) % 1)*len(classes)),' --> ',c)
        plt.plot(d2[i,0],d2[i,1],marker="o",color=c,linestyle="None")  
        i+=1  
    plt.xlabel('1st Component')  
    plt.ylabel('2nd Component')  
    plt.title(n2+' PCA Components')
    plt.subplot(224)
    i=0
    for l in l1:
        c_i=nm.where(classes==l)[0]
        c=colors[int((c_i/len(classes) % 1)*len(classes))]
        plt.plot(d2[i,1],d2[i,2],marker="o",color=c,linestyle="None")
        i+=1
    plt.xlabel('2nd Component')  
    plt.ylabel('3rd Component')   
    plt.savefig(f)