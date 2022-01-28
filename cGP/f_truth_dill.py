import numpy as np
##################################################
#Example: Use a previously fitted surrogate as black-box function    #
##################################################
EXAMPLE_NAME='dill.load'
#This loads the black-box function
#
########################################
#          Function wrapping           #
########################################
#The black-box function would return the observed response value Y' for X'. 
#This wrapping would makes the black-box function to be piece-wise constant. 
#
#from scipy.spatial.distance import cdist
#import pickle as pkl
import dill as pkl
with open('BUKIN.pkl', 'rb') as surrogate_file:
	pkl_dict = pkl.load(surrogate_file)
print(pkl_dict)
def f_truth(X):
    my_X = X.reshape(1,-1)
    clf_XY = pkl_dict['Classify']
    dgm_XY = pkl_dict['Cluster']
    my_X_label = clf_XY.predict(my_X)
    mt = pkl_dict['GPR_list'][int(my_X_label)]
    #if USE_SKLEARN:
    my_gp = mt.predict(my_X, return_std=True, return_cov=False)
    my_mu = my_gp[0]
    my_sigma = my_gp[1]
    return my_mu[0,0]
print(f_truth(np.asarray([1,1]).reshape(1,-1)))

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
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 9, jac='2-point', hess=BFGS())
#The problem here is that we cannot obtain the higher order derivatives of my_obj in general, we use approximations with 2-point difference and BFGS/SR1 method to get a numerical supplier. 

########################################
#       Constraints on response Y      #
########################################
def censor_function(Y):
    #return Y #if you don't want any censor, use this line as the definition of your censor function.
    ret = Y
    return ret

