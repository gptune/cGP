import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS

##################################################
# Robot arm function as the emulated function    #
##################################################
EXAMPLE_NAME='ROBOT_ARM'

########################################
#          Function wrapping           #
########################################
# This wraps the robot arm function to be used by the optimizer.
#
def robot_arm_function(theta, L):
    print('L',L)
    u = sum(L[i] * np.cos(sum(theta[:i + 1])) for i in range(4))
    v = sum(L[i] * np.sin(sum(theta[:i + 1])) for i in range(4))
    return np.sqrt(u**2 + v**2)

def f_truth(X):
    X = X.reshape(1,-1)
    # Here, we assume that the input X is a flat array containing
    # 4 theta values followed by 4 L values
    theta = X[0,:4]  # Extract theta values
    L = X[0,4:]     # Extract L values
    print(theta,L,X)
    # Normalize theta values to be within [0, 2*pi]
    theta = np.mod(theta, 2*np.pi)
    
    # Call the robot arm function
    return -robot_arm_function(theta, L)  # We negate the result for minimization

# Example usage of f_truth
example_parameters = np.asarray([np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 0.5, 0.5, 0.5, 0.5])
print(f"Example robot arm calculation: {f_truth(example_parameters)}")

########################################
#       Soft Constraints on input X    #
########################################
def boundary_penalty(X, data_X=None):
    return 0  # No additional penalty in this example

########################################
#       Hard Constraints on input X    #
########################################
def get_bounds(restrict):
    bds = np.array([
        [0, 2*np.pi],  # theta1
        [0, 2*np.pi],  # theta2
        [0, 2*np.pi],  # theta3
        [0, 2*np.pi],  # theta4
        [0, 1],       # L1
        [0, 1],       # L2
        [0, 1],       # L3
        [0, 1]        # L4
    ]).astype(float)
    return bds

bounds = get_bounds(1)
lw = bounds[:,0].tolist()
up = bounds[:,1].tolist()

# The bound constraints are defined using a Bounds object.
bounds_constraint = Bounds(lw, up)

# The linear constraints are defined using a LinearConstraint object.
linear_constraint = LinearConstraint([[0]*8], [-np.inf], [np.inf])

# The nonlinear constraints are defined using a NonlinearConstraint object.
def cons_f(x):
    return 0  # No additional constraints in this example
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, np.inf, jac='2-point', hess=BFGS())

########################################
#       Constraints on response Y      #
########################################
def censor_function(Y):
    # No censoring in this example
    return Y

