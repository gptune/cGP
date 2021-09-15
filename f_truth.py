import numpy as np
##################################################
#   Example: 2-dimensional non-smooth function   #
##################################################
EXAMPLE_NAME='BUKIN_N6'
#This loads the black-box function
#
########################################
#          Function wrapping           #
########################################
#The black-box function would return the observed response value Y' for X'. 
#This wrapping would makes the black-box function to be piece-wise constant. 
#
#from scipy.spatial.distance import cdist
def f_truth(X):
    X = X.reshape(1,-1)
    ret = 100*np.sqrt(np.abs(X[:,1]-0.01*X[:,0]**2))
    ret = ret + 0.01*np.abs(X[:,0]+10)
    return ret[0]
print(f_truth(np.asarray([0.1,0.9]).reshape(1,-1)))

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
    bds = np.array([[-15,-5],[-3,3]]).astype(float)
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
linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, -np.inf], [np.inf, np.inf])

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

