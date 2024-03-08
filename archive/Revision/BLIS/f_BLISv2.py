import numpy as np
##################################################
#   Example: 2-dimensional non-smooth function   #
##################################################
EXAMPLE_NAME='BLISv2'
#This loads the black-box function
#
########################################
#          Function wrapping           #
########################################
#The black-box function would return the observed response value Y' for X'. 
#This wrapping would makes the black-box function to be piece-wise constant. 
#
#from scipy.spatial.distance import cdist
import pandas as pd
from scipy.interpolate import griddata

# Load the data
df = pd.read_csv('/media/hrluo/WORK1/LBNLhrluo/minimalcGP/output-jobs-and-subjobs-301-304.csv')

# Prepare the data for interpolation
points = df[['mc', 'nc', 'kc']].values  # The grid points where we have data
values = df['min_jobs(min_subjobs(min_sec)'].values  # The corresponding time values

# Define f_truth to interpolate the data based on mc, nc, kc
def f_truth(X):
    # Assuming X is a single array-like object containing [mc, nc, kc]
    X = X.squeeze()
    print(X)
    mc, kc, nc = X[0], X[1], X[2]
    
    # Round mc, nc, kc to the nearest multiple of 8
    mc = np.round(X[0] / 12) * 12
    kc = np.round(X[1] / 16) * 16
    nc = 4080#np.round(X[2] / 6) * 6
    # Ensure mc, nc, kc are not rounded down to zero if they were originally non-zero
    #mc = max(mc, 8) if X[0] > 0 else 0
    #nc = max(nc, 8) if X[1] > 0 else 0
    #kc = max(kc, 8) if X[2] > 0 else 0
    
    # Find the row in the DataFrame where mc, nc, kc match
    row = df[(df['mc'] == mc) & (df['nc'] == nc) & (df['kc'] == kc)]
    
    # If there's no exact match, return 1
    if not row.empty:
        #return np.array([row['time'].values[0]])
        return np.array([row['min_jobs(min_subjobs(min_sec)'].values[0]])
    else:
        return np.array([1.])  # Return 1 if the combination is not found

# Example usage of f_truth
mc_val = 72.0  # Example value, replace with actual query
nc_val = 256.0   # Example value, replace with actual query
kc_val = 8. # Example value, replace with actual query
time_output = f_truth(np.array([mc_val, nc_val, kc_val]) )
print(f"Time output for mc={mc_val}, nc={nc_val}, kc={kc_val}: {time_output}")



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
    bds = np.array([[1,2002],[1,2002],[4079,4081]]).astype(float)
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
linear_constraint = LinearConstraint([[0, 0, 0]], [-np.inf], [np.inf])

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

