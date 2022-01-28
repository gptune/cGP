import numpy as np
from scipy.stats import norm
#Define the expected improvement as objective function to optimize over.
def expected_improvement(X,surrogate,X_sample,Y_sample,correct_label,classify,USE_SKLEARN=True,VERBOSE=True,boundary_penalty=lambda a,b: 0):
    my_X = np.copy(X)
    my_X = X.reshape(1, -1)
    my_X_label = classify.predict(my_X)
    #If not in this component, set it to zero immediately.
    if my_X_label != correct_label.astype(int): return -0
    my_xi = 0.0 #tuning parameter, set it to zero for now.
    if USE_SKLEARN:
        my_gp = surrogate.predict(my_X, return_std=True, return_cov=False)
        my_mu = my_gp[0]
        my_sigma = my_gp[1]
    else:
        my_gp = surrogate.predict(my_X)
        my_mu = my_gp[0]
        my_sigma = my_gp[1]
        my_sigma = np.sqrt(np.absolute(my_sigma)).reshape(-1, 1)
    my_mu = np.asarray(my_mu)
    my_sigma = np.asarray(my_sigma)
    with np.errstate(divide='warn'):
        my_imp = my_mu - np.max(Y_sample) - my_xi
        my_Z = np.divide(my_imp,my_sigma)
        #norm = mvn(mean=np.zeros(X_sample[0,:].shape), cov=np.eye(X_sample.shape[1]))
        my_ei = my_imp * norm.cdf(my_Z) + my_sigma * norm.pdf(my_Z)
        my_ei[np.where(my_sigma <= 0.0)] = 0.0
    #Here we penalize the acquisition function value according to boundary_penalty function, by default this would be disabled. See document for details.
    my_ei = my_ei + boundary_penalty(my_X,X_sample)
    my_ei = float(my_ei.ravel())
    if VERBOSE: print('EI=',my_ei,'\n')
    return - my_ei/X_sample.shape[0] #We want to minimize this quantity. maximizing expected improvement
    
def mean_square_prediction_error(X,surrogate,X_sample,Y_sample,correct_label,classify,USE_SKLEARN=True,VERBOSE=True,boundary_penalty=lambda a,b: 0):
    my_X = np.copy(X)
    my_X = X.reshape(1, -1)
    my_X_label = classify.predict(my_X)
    #If not in this component, set it to zero immediately.
    if my_X_label != correct_label.astype(int): return -0
    my_xi = 0.0 #tuning parameter, set it to zero for now.
    if USE_SKLEARN:
        #my_gp = surrogate.predict(my_X, return_std=False, return_cov=True)
        #my_mu = my_gp[0]
        #my_sigma = my_gp[1]
        #my_gp_obs = surrogate.predict(X_sample, return_std=False, return_cov=True)
        #my_mu_obs = my_gp_obs[0]
        #my_sigma_obs = my_gp_obs[1]
        X_joint = np.vstack((my_X,X_sample))
        mu_cross, sigma_cross = surrogate.predict(X_joint, return_std=False, return_cov=True)
        #print('\n',sigma_cross.shape,'>>>',my_X.shape[0])
        sigma = sigma_cross[0:my_X.shape[0],0:my_X.shape[0]]
        sigma_obs = sigma_cross[my_X.shape[0]:sigma_cross.shape[1],my_X.shape[0]:sigma_cross.shape[1]]
        sigma_cross = sigma_cross[0:my_X.shape[0],my_X.shape[0]:sigma_cross.shape[1]]
        sigma_cross = sigma_cross.reshape(-1,my_X.shape[0]).T
    else:
        #my_gp = surrogate.predict(my_X,full_cov=True)
        #my_mu = my_gp[0]
        #my_sigma = my_gp[1]
        #my_gp_obs = surrogate.predict(X_sample,full_cov=True)
        #my_mu_obs = my_gp_obs[0]
        #my_sigma_obs = my_gp_obs[1]
        X_joint = np.vstack((my_X,X_sample))
        mu_cross, sigma_cross = surrogate.predict(X_joint, full_cov=True)
        sigma = sigma_cross[0:my_X.shape[0],0:my_X.shape[0]]
        sigma_obs = sigma_cross[my_X.shape[0]:sigma_cross.shape[1],my_X.shape[0]:sigma_cross.shape[1]]
        sigma_cross = sigma_cross[0:my_X.shape[0],my_X.shape[0]:sigma_cross.shape[1]]
        sigma_cross = sigma_cross.reshape(-1,my_X.shape[0]).T
    mspe = sigma - sigma_cross @ sigma_obs @ sigma_cross.T + my_xi
    if mspe.shape[0]>1:
        mspe = np.diag(mspe)
    else:
        mspe = float(mspe.ravel())    
    mspe = mspe + boundary_penalty(my_X,X_sample)
    if VERBOSE: print('MSPE=',mspe,'\n')
    return mspe/X_sample.shape[0] #We want to minimize this quantity. minimizing mspe

