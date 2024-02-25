import numpy as np
##################################################
#   Example: 2-dimensional non-smooth function   #
##################################################
EXAMPLE_NAME='hypre2d_dimmerge'
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
df = pd.read_csv('/home/hrluo/gptune_cgp/GPTune/examples/Hypre/hypre2d_dimmerge_grid_data.csv')

# Prepare the data for interpolation
points = df['x'].values  # The grid points where we have data
values = df['y'].values  # The corresponding time values

# Define f_truth to interpolate the data based on mc, nc, kc
def f_truth(X):
    # X is expected to be an array-like object with dimensions [x_1, x_2, ..., x_n]
    X = np.array(X).squeeze()  # Ensure X is a 1D array

    # Calculate the Euclidean distance between X and each point in the points array
    distances = np.abs(points - X)

    # Find the index of the closest point
    closest_index = np.argmin(distances)

    # Return the 'value' corresponding to the closest 'point'
    return values[closest_index]

 


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
    bds = np.array([[0,4]]).astype(float)
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
linear_constraint = LinearConstraint([[0], [-np.inf], [np.inf]])

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

