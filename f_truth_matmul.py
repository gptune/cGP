import numpy as np
##################################################
#   Example: 2-dimensional non-smooth function   #
##################################################
matrix_size = 1000
EXAMPLE_NAME='MATMUL_'+str(matrix_size)
#This loads the black-box function
from numpy import genfromtxt
my_data = genfromtxt('Giulia_'+str(matrix_size)+'.csv', delimiter=',')

my_data = np.delete(my_data, (0), axis=0)
my_data = np.delete(my_data, (0), axis=1)
print(my_data)
Y_obs = my_data[:,1].astype(float).reshape(-1,1)
X_obs = my_data[:,0].astype(float).reshape(-1,1)
#Dimension of the input domain
#d = X_obs.shape[1]
print(X_obs.shape)
print(Y_obs.shape)
#
########################################
#          Function wrapping           #
########################################
#The black-box function would return the observed response value Y' for X'. 
#This wrapping would makes the black-box function to be piece-wise constant. 
#
from scipy.spatial.distance import cdist
def f_truth(X):
    X = np.round(X)
    to_obs = cdist(np.round(X),X_obs, metric='euclidean')
    closest_obs = np.argmin(to_obs)
    ret_X = X_obs[closest_obs,:]
    ret_Y = Y_obs[closest_obs,:]
    ret_X = int(X)
    #print(np.argwhere(ret_X==X_obs))
    #ret_Y = Y_obs[np.argwhere(ret_X==X_obs)[0,0],:]
    ret_Y = Y_obs[np.argmin(np.abs(ret_X-X_obs) ),:]
    print('Closest point in dataset is ',ret_X,' with observed value ',ret_Y[0])
    return ret_Y[0].astype(float)
point1 = np.ones((1,1))*256.0
print(f_truth(point1))

########################################
#       Soft Constraints on input X    #
########################################
#This sets up penalty functions on the acquisition function.
def boundary_penalty(X,data_X=None):
    return 0

########################################
#       Hard Constraints on input X    #
########################################
#This sets up the domain over which the acquisition function is maximized, and also shared by random search.
def get_bounds(restrict):
    #if restrict == 1:
    bds = np.array([[1,matrix_size]]).astype(float)
    return bds
    
bounds = get_bounds(1)
lw = bounds[:,0].tolist()
up = bounds[:,1].tolist()

#The bound constraints are defined using a Bounds object.
from scipy.optimize import Bounds
bounds_constraint = Bounds(lw, up)

#The linear constraints are defined using a LinearConstraint object.
#The constraints here is x_0+2x_1<=1 and  2x_0+x_1=1
from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint([[1], [1]], [-np.inf, -np.inf], [np.inf, np.inf])

#The non-linear constraints are defined using a NonlinearConstraint object.
#The constraints here is -inf<=x_0^2+x_1<=1 and  -inf<=x_0^2-x_1<=1
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
def cons_f(x):
    return 0
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, np.inf, jac='2-point', hess=BFGS())
#The problem here is that we cannot obtain the higher order derivatives of my_obj in general, we use approximations with 2-point difference and BFGS/SR1 method to get a numerical supplier. 

########################################
#       Constraints on response Y      #
########################################
def censor_function(Y):
    #return Y #if you don't want any censor, use this line as the definition of your censor function.
    ret = Y
    return ret

