import numpy as np
##################################################
#   Example 4: record dataset mapping, matmul    #
##################################################
EXAMPLE_NAME='matmul'
#This loads the dataset for building a black-box function
#The format of the dataset should be a csv file, the first column being the response (Y)
#The rest columns of the dataset is the d-dimensional inputs (X)
#
def get_bounds(restrict):
    if restrict == 1:
        bounds1 = np.array([[10,4096]]).astype(float)
    else:
        bounds1 = np.array([[1,4096]]).astype(float)
    #print(bounds)
    return bounds1

from numpy import genfromtxt
my_data = genfromtxt('haswell_5.csv', delimiter=',')

Y_obs = my_data[:,2].astype(float).reshape(-1,1)
X_obs = my_data[:,1].astype(float).reshape(-1,1)
#Dimension of the input domain
#d = X_obs.shape[1]
print(X_obs)
print(Y_obs)

########################################
#          Function wrapping           #
########################################
#This allows us to wrap a real-world dataset into the format of a black-box function useful 
#Given a point X, we find the closest point X' in the dataset (by some distance measure, currently L^2).
#The black-box function would return the observed response value Y' for X'. 
#This wrapping would makes the black-box function to be piece-wise constant. 
#
from scipy.spatial.distance import cdist
def f_truth(X):
    to_obs = cdist(X,X_obs, metric='euclidean')
    closest_obs = np.argmin(to_obs)
    ret_X = X_obs[closest_obs,:]
    ret_Y = Y_obs[closest_obs,:]
    ret_X = int(X)
    #print(np.argwhere(ret_X==X_obs))
    ret_Y = Y_obs[np.argwhere(ret_X==X_obs)[0,0],:]
    print('Closest point in dataset is ',ret_X,' with observed value ',ret_Y[0])
    return ret_Y[0].astype(float)

#point1 = np.ones((1,1))*256.0
#print(f_truth(point1))

def boundary_penalty(X,data_X=None):
    #return 0
    #return np.zeros((X.shape[0],1)) #if you don't want any penalty, use this line as the definition of your penalty
    #ret = []
    #for g in range(X.shape[0]):
    #    g_list = []
    #    for h in range(bounds.shape[1]):
    #        g_list.append( np.sum( (X[g,:]-bounds[:,h])**2 ) )
    #    ret.append(min(g_list))
    #res = X.astype(int)%8==0
    #return res*(100)\
    #if X<100:
    #    return -1e5
    if X.astype(int)%8==0:
        return 0
    else:
        return -1e3
    return -1e3
def censor_function(Y):
    #return Y #if you don't want any censor, use this line as the definition of your censor function.
    ret = Y
    #ret = Y.*(Y<20000 & Y>100)
    return ret#-np.minimum(0.1,10/np.asarray(ret))
#ver 0.6 new, 
#if random_domain returns TRUE, then such a choice by the random step is acceptable.
#if random_domain returns FALSE, then such a choice is out of our search input domain, and we would like to re-sample another random location.
def random_domain(X,data_X=None,REPEAT_SAMPLE=False):
    #return True
    #REPEAT_SAMPLE = False
    for i in range(data_X.shape[0]):
        if all(X.astype(int)== data_X[i,:].astype(int)) and ~REPEAT_SAMPLE: return False
        #This is only for matmul example searching only multiples of 8.
    return X.astype(int)%8==0 

