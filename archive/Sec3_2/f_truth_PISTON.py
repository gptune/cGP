import numpy as np
##################################################
#    Example: 7-dimensional emulated function    #
##################################################
EXAMPLE_NAME='PISTON'
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
    M = X[:,0]
    S = X[:,1]
    V0= X[:,2]
    k = X[:,3]
    P0= X[:,4]
    Ta= X[:,5]
    T0= X[:,6]
    A = P0*S+19.62*M-k*V0/S
    V = (S/2*k)*(np.sqrt(A*A+4*k*(P0*V0/T0)*Ta )-A)
    C = 2*np.pi*np.sqrt( M/(k+S*S*(P0*V0/T0)*(Ta/(V*V)) ) )
    ret = C
    return ret[0]
print(f_truth(np.asarray([40,0.010,0.005,2500,100000,294,350]).reshape(1,-1)))

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
    bds = np.array([[30,60],[0.005,0.020],[0.002,0.010],[1000,5000],[90000,110000],[290,296],[340,360]]).astype(float)
    return bds

bounds = get_bounds(1)
lw = bounds[:,0].tolist()
up = bounds[:,1].tolist()

#The bound constraints are defined using a Bounds object.
from scipy.optimize import Bounds
bounds_constraint = Bounds(lw, up)

#The linear constraints are defined using a LinearConstraint object.
from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint([[1,1,1,1,1,1,1]], [-np.inf], [np.inf])

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
    return ret#-np.minimum(0.1,10/np.asarray(ret))
