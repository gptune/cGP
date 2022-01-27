#!/usr/bin/env python
# coding: utf-8
########################################
#     clustered Gaussian surrogate     #
########################################
#Author: Hengrui Luo
#hrluo@lbl.gov
#Modified by: Yang Liu
#Last update: 2022-Jan-25
########################################

import sys
# sys.modules[__name__].__dict__.clear()
# import sys
#Warnings supression
import warnings
warnings.filterwarnings('ignore')
print("Python version: ", sys.version)
# print(sys.argv)

#Print the numpy version and set the random seed
import numpy as np
print('numpy version: ', np.__version__)
from numpy import int64	
from numpy import int	
from numpy import float	
from numpy import bool


#Get a random string stamp for this specific run, used for the filename of image export.
import random
import string

#Print the GPy version
import GPy
print('GPy version: ', GPy.__version__)

#Print the sklearn version
import sklearn
print('sklearn version: ', sklearn.__version__)
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import LinearConstraint

from sklearn.gaussian_process import *
from datetime import datetime
import acquisition as acq

from scipy import stats
from scipy.optimize import SR1
from scipy.optimize import minimize


class cGP_constrained(object):
    """
    This is the cGP_constrained callable interface, in addition to the command-line interface cGP_constrained.py 
	"""
    def __init__(self, DP, DO, **kwargs):
        self.DP = DP 
        self.DO = DO 

        if "N_PILOT" in kwargs:
            self.N_PILOT=kwargs['N_PILOT']
        else:
            self.N_PILOT=20

        if "N_SEQUENTIAL" in kwargs:
            self.N_SEQUENTIAL=kwargs['N_SEQUENTIAL']
        else:
            self.N_SEQUENTIAL=20                

        if "RND_SEED" in kwargs:
            self.RND_SEED=kwargs['RND_SEED']
        else:
            self.RND_SEED=1  

        if "EXAMPLE_NAME" in kwargs:
            self.EXAMPLE_NAME=kwargs['EXAMPLE_NAME']
        else:
            self.EXAMPLE_NAME='default_name'


#Which method should we use for the Bayesian optimization scheme?
#'FREQUENTIST' method means that the (hyper-)parameters are estimated by using some frequestist optimization like lbfgs.
#'BAYESIAN' method means that the paramteres are estimated by putting a prior(Gamma)-posterior mechnism, the estimated value would be posterior mean.
        if "METHOD" in kwargs:
            self.METHOD=kwargs['METHOD']  
        else:
            self.METHOD='FREQUENTIST'       

#Following 3 parameters are only for HMC Bayesian sampling, you have to choose METHOD  = 'BAYESIAN' to use these parameters.
        if "N_BURNIN" in kwargs:
            self.N_BURNIN=kwargs['N_BURNIN']  
        else:
            self.N_BURNIN=500
        if "N_MCMCSAMPLES" in kwargs:
            self.N_MCMCSAMPLES=kwargs['N_MCMCSAMPLES']  
        else:
            self.N_MCMCSAMPLES=500
        if "N_INFERENCE" in kwargs:
            self.N_INFERENCE=kwargs['N_INFERENCE']  
        else:
            self.N_INFERENCE=500

#Exploration rate is the probability (between 0 and 1) of following the next step produced by acquisition function.
        if "EXPLORATION_RATE" in kwargs:
            self.EXPLORATION_RATE=kwargs['EXPLORATION_RATE']  
        else:
            self.EXPLORATION_RATE=1.0    

#Do you want a cluster GP? If NO_CLUSTER = True, a simple GP will be used.
        if "NO_CLUSTER" in kwargs:
            self.NO_CLUSTER=kwargs['NO_CLUSTER']  
        else:
            self.NO_CLUSTER=False

#Do you want to amplify the weight/role of response X when doing clustering?
        self.X_AMPLIFY = 1.
#Do you want to subtract an amount from the response X when doing clustering?
        self.X_TRANSLATE = []
#Do you want to amplify the weight/role of response Y when doing clustering?
        self.Y_AMPLIFY = 1.
#Do you want to subtract an amount from the response Y when doing clustering?
        self.Y_TRANSLATE = 0.
#What is the maximal number of cluster by your guess? This option will be used only if NO_CLUSTER=False.

#When deciding cluster components, how many neighbors shall we look into and get their votes? This option will be used only if NO_CLUSTER=False.
        if "N_NEIGHBORS" in kwargs:
            self.N_NEIGHBORS=kwargs['N_NEIGHBORS']  
        else:
            self.N_NEIGHBORS=3


#Cluster method: BGM or KMeans
        if "CLUSTER_METHOD" in kwargs:
            self.CLUSTER_METHOD=kwargs['CLUSTER_METHOD']  
        else:
            self.CLUSTER_METHOD='BGM'

#What is the maximal number of cluster by your guess? This option will be used only if NO_CLUSTER=False.
        if "N_COMPONENTS" in kwargs:
            self.N_COMPONENTS=kwargs['N_COMPONENTS']  
        else:
            self.N_COMPONENTS=3

#This is a GPy parameter, whether you want to normalize the response before/after fitting. Don't change unless necessary.
        self.GPy_normalizer = True
#Whether we should sample repetitive locations in the sequential sampling procedure.
#If True, we would keep identical sequential samples no matter what. (Preferred if we believe a lot of noise)
#If False, we would re-sample when we run into identical sequential samples. (Default)
#In a acquisition maximization step, this is achieved by setting the acquisition function at repetitive samples to -Inf
#In a random search step, this is achieved by repeat the random selection until we got a new location.
        self.REPEAT_SAMPLE = False

        self.USE_SKLEARN = True
        self.ALPHA_SKLEARN = 1e-5
#Value added to the diagonal of the kernel matrix during fitting. 
        self.SKLEARN_normalizer = True

#acquisition function: EI or MSPE
        if "ACQUISITION" in kwargs:
            self.ACQUISITION=kwargs['ACQUISITION']  
        else:
            self.ACQUISITION='EI'

    def f_Classify(self):
        return KNeighborsClassifier(self.N_NEIGHBORS)

    def f_Cluster(self,RND_SEED):
        if(self.CLUSTER_METHOD=='KMeans'):
            return KMeans(n_clusters=N_COMPONENTS, 
                           random_state=RND_SEED)
        if(self.CLUSTER_METHOD=='BGM'):
            return BayesianGaussianMixture(
                            #weight_concentration_prior_type="dirichlet_distribution",
                            weight_concentration_prior_type="dirichlet_process",
                            n_components=self.N_COMPONENTS,#pick a big number, DGM will automatically adjust
                            random_state=RND_SEED)

    ########################################
    #       Soft Constraints on input X    #
    ########################################
    #This sets up penalty functions on the acquisition function.
    def boundary_penalty(self,X,data_X=None):
        return 0

    ########################################
    #       Hard Constraints on input X    #
    ########################################
    #This sets up the domain over which the acquisition function is maximized, and also shared by random search.
    def get_bounds(self,restrict):
        #if restrict == 1:
        bds = np.array([[-1,1],[-1,1]]).astype(float)
        return bds
    
    #The non-linear constraints are defined using a NonlinearConstraint object.
    #The constraints here is -inf<=x_0^2+x_1<=1 and  -inf<=x_0^2-x_1<=1    
    def get_nonlinear_constraint(self):
        def cons_f(x):
            return 0
        nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, np.inf, jac='2-point', hess=BFGS())
        return nonlinear_constraint
    #The problem here is that we cannot obtain the higher order derivatives of my_obj in general, we use approximations with 2-point difference and BFGS/SR1 method to get a numerical supplier. 


    ########################################
    #       Constraints on response Y      #
    ########################################
    def censor_function(self,Y):
        #return Y #if you don't want any censor, use this line as the definition of your censor function.
        ret = Y
        return ret

    #The linear constraints are defined using a LinearConstraint object.
    #The constraints here is x_0+2x_1<=1 and  2x_0+x_1=1   
    def get_linear_constraint(self):
        linear_constraint = LinearConstraint([[0]*self.DP], [-np.inf], [np.inf])
        return linear_constraint

    #The acquisition function
    def get_acquisition(self):
        if(self.ACQUISITION=='EI'):
            return acq.expected_improvement
        if(self.ACQUISITION=='MSPE'):
            return acq.mean_square_prediction_error


    def get_random_string(self,length):
        return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


    #This recode function will turn the labels into increasing order,e.g. [1, 1, 3, 3, 0] ==> [0, 0, 1, 1, 2].
    def recode(self,label):
        level = np.unique(np.array(label))
        ck = 0
        for j in level:
            label[label==j]=ck
            ck=ck+1
        return label


    def f_truth(self,X):
        X = X.reshape(1,-1)
        x=X[:,0]
        y=X[:,1]
        return( 1/(1 + (x - .25)**2 + (y - .25)**2) )


    def run(self):

        RND_SEED = self.RND_SEED
        np.random.seed(RND_SEED)
        print('Seed=',RND_SEED)

        print('start cGP_constrained:')

        rdstr=self.get_random_string(8)
        print('random stamp for this run:',rdstr)


        acq_fun = self.get_acquisition()
        clf_XY = self.f_Classify()
        dgm_XY = self.f_Cluster(self.RND_SEED)


        bounds = self.get_bounds(1)
        lw = bounds[:,0].tolist()
        up = bounds[:,1].tolist()
        #The bound constraints are defined using a Bounds object.
        bounds_constraint = Bounds(lw, up)
        print('bounds',bounds.shape)
        inp_dim=bounds.shape[0]
        
        linear_constraint = self.get_linear_constraint()
        nonlinear_constraint = self.get_nonlinear_constraint()

        ########################################
        #         Kernel specification         #
        ########################################
        #Which kernel you want to use for your model? Such a kernel must be implemented as a GPy/sklearn kernel class.
        if self.USE_SKLEARN==True:
            KERNEL_TEMPLATE = sklearn.gaussian_process.kernels.Matern(length_scale=np.ones(inp_dim,), length_scale_bounds=(1e-05, 100000.0), nu=3/2) + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05, 100000.0))
            #KERNEL_TEMPLATE = sklearn.gaussian_process.kernels.Matern(length_scale=np.ones(inp_dim,), length_scale_bounds=(1e-05, 100000.0), nu=1/2)
        else:
            KERNEL_TEMPLATE = GPy.kern.Matern32(input_dim=inp_dim, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=inp_dim)
            #KERNEL_TEMPLATE = GPy.kern.Exponential(input_dim=inp_dim, variance=1., lengthscale=1.)
        #Do you want to penalize boundary sample points? If so, how?

        # datetime object containing current date and time
        samplestartingtime = datetime.now()
        ########################################
        #          Draw pilot samples          #
        ########################################
        #This cell only provides a pilot sample.
        #Prepare pilot samples (X,Y)
        print('Example : ',self.EXAMPLE_NAME)
        print('\n>>>>>>>>>>Sampling ',self.N_PILOT,' pilot samples...<<<<<<<<<<\n')
        X_sample = np.zeros((self.N_PILOT,bounds.shape[0]))
        for j in range(bounds.shape[0]):
                X_sample[:,j] = np.random.uniform(bounds[j,0],bounds[j,1],size=(self.N_PILOT,1)).ravel()

        Y_sample = np.zeros((self.N_PILOT,1))
        Y_sample = np.zeros((self.N_PILOT,1))
        for k in range(self.N_PILOT):
            Y_sample[k,0] = self.f_truth(X_sample[k,:].reshape(1,-1))
            Y_sample[k,0] = self.censor_function(Y_sample[k,0])
        #print('Pilot X',X_sample)
        #print('Pilot Y',Y_sample)


        #The cGP procedure consists of following steps
        #Step 1. For observations, we can do a (unsupervised) (X,Y)-clustering and label them, different components are generated.
        #Step 2. For predictive locations, we can do a (supervised) k-nearest neighbor classification, and predict at each location based on which component it belongs to. 
        #Step 3. We compute the acquisition function and then proceed to the next sample, after adding the new sample we repeat Step 1 and 2.

        #Prepare an up-to-date X_TRANSLATE, as the empirical mean of the X_sample
        if len(self.X_TRANSLATE)>0:
            self.X_TRANSLATE = np.mean(X_sample,axis=0)
        else:
            self.X_TRANSLATE = np.mean(X_sample,axis=0)*0
        #Prepare an up-to-date Y_TRANSLATE, as the empirical mean of the Y_sample
        if self.Y_TRANSLATE != 0:
            self.Y_TRANSLATE = np.mean(Y_sample)
        #print(Y_sample - Y_TRANSLATE)
        print(np.concatenate((X_sample,self.Y_AMPLIFY*(Y_sample-self.Y_TRANSLATE)),axis=1))
        #Prepare initial clusters, with XY-joint.
        XY_sample = np.concatenate((self.X_AMPLIFY*(X_sample-self.X_TRANSLATE),self.Y_AMPLIFY*(Y_sample-self.Y_TRANSLATE)),axis=1)
        XY_label  = dgm_XY.fit_predict(XY_sample)
        print('\n Initial labels for (X,Y)-joint clustering',XY_label)
        #Make copies of X_sample for X-only fitting and XY-joint fitting.
        X_sample_XY = np.copy(X_sample)
        Y_sample_XY = np.copy(Y_sample)
        #Prepare initial labels
        clf_XY.fit(X_sample_XY, XY_label)
        #This is an artifact, we need to have at least d samples to fit a d-dimensional GP model (for its mean and variance)
        for c in np.unique(XY_label):
            if sum(XY_label==c)<=X_sample_XY.shape[1]+2:
                occ = np.bincount(XY_label)
                XY_label[np.where(XY_label==c)] = np.argmax(occ)


        ########################################
        #        Draw sequential samples       #
        ########################################
        VERBOSE = False
        #Prepare sequential samples (X,Y)
        print('\n>>>>>>>>>>Sampling ',self.N_SEQUENTIAL,' sequential samples...<<<<<<<<<<\n')
        X_sample = X_sample_XY
        Y_sample = Y_sample_XY
        cluster_label = XY_label


            
        #Main loop that guides us in sampling sequential samples
        component_label = np.unique(np.array(cluster_label))

        for it in range(self.N_SEQUENTIAL):
            print('\n>>>>>>>>>> ***** STEP ',it+1,'/',self.N_SEQUENTIAL,'***** <<<<<<<<<<')
            print('\n>>>>>>>>>> +++++ N_PROC disabled +++++ <<<<<<<<<<')
            #Step 1. For observations, we can do a (unsupervised) (X,Y)-clustering and label them, different components are generated.
            #Create the (X,Y) joint sample to conduct (unsupervised clustering)
            #if len(self.X_TRANSLATE)>0:
            #    self.X_TRANSLATE = np.mean(X_sample,axis=0)
            #if self.Y_TRANSLATE != 0:
            #    self.Y_TRANSLATE = np.mean(Y_sample)
            #The cluster must be based on adjusted response value Y.
            XY_sample        = np.concatenate((self.X_AMPLIFY*(X_sample-self.X_TRANSLATE),self.Y_AMPLIFY*(Y_sample-self.Y_TRANSLATE).reshape(-1,1)),axis=1)
            if self.NO_CLUSTER:
                print('>>NO CLUSTER, a GP surrogate.')
                cluster_label    = np.zeros(XY_sample.shape[0])
            else:
                #print('>>CLUSTERED, a cGP surrogate. ',len(component_label),' components in surrogate model.')
                cluster_label    = dgm_XY.fit_predict(XY_sample)#cluster_label
                if VERBOSE: print('dgm label', cluster_label)
                #Again, we need to ensure that every cluster has at least d (dimension of covariate) samples.
                for c in np.unique(cluster_label):
                    if sum(cluster_label==c)<=X_sample.shape[1]+2:
                        occ = np.bincount(cluster_label)
                        cluster_label[np.where(cluster_label==c)] = np.argmax(occ)
                if VERBOSE: print('merged label',cluster_label)
            cluster_label = self.recode(cluster_label)
            if VERBOSE: print('All labels are recoded: ',cluster_label)
                    
            #Step 2. For predictive locations, we can do a (supervised) k-nearest neighbor classification, and predict at each location based on which component it belongs to. 
            #Refresh the Classifier
            clf_XY.fit(X_sample,cluster_label)
                
            #Step 3. We either randomly search one location or compute the acquisition function and then proceed to the next sample, after adding the new sample we repeat Step 1 and 2.
            coin = np.random.uniform(0,1,1)
            if coin<self.EXPLORATION_RATE:
                component_label = np.unique(np.array(cluster_label))
                if not self.NO_CLUSTER:
                    print('>>CLUSTERED, a cGP surrogate.',len(component_label),' components in surrogate model.');    
                print('>>>>Find next sample: acquisition proposal.')
                opt_x = np.zeros((1,X_sample.shape[1]))
                opt_acq = - np.inf
                object_model_list = [None]*len(component_label)
                for c in component_label:
                    #Assign the corresponding X_sample and Y_sample values to the cluster coded by c. 
                    c_idx = np.where(cluster_label == int(c))
                    if VERBOSE: 
                        print('>>>>Fitting component ',c,'/',len(component_label)-1,' total components')
                        print(c_idx)
                    Xt = X_sample[c_idx].ravel().reshape(-1,X_sample.shape[1])
                    Yt = Y_sample[c_idx].ravel().reshape(-1,1)
                    #Fit the model with normalization
                    if self.USE_SKLEARN==True:
                        mt = GaussianProcessRegressor(kernel=KERNEL_TEMPLATE, random_state=0, normalize_y=self.SKLEARN_normalizer,alpha=self.ALPHA_SKLEARN,  
                                                    optimizer='fmin_l_bfgs_b', n_restarts_optimizer=int(10*bounds.shape[0]))
                    else:
                        mt = GPy.models.GPRegression(Xt, Yt, kernel=KERNEL_TEMPLATE, normalizer=self.GPy_normalizer)
                    ###
                    if self.METHOD == 'FREQUENTIST':
                        ##############################
                        #Frequentist MLE GP surrogate#
                        ##############################
                        print('>>>>>>METHOD: frequentist MLE approach, component '+str(c)+'/'+str(len(component_label)-1))
                        print('>>>>>>SAMPLE: component sample size =',len(c_idx[0]) )
                        if self.USE_SKLEARN==True:
                            mt.fit(Xt, Yt)
                            #No need to do more for sklearn GP
                            print('>>>>>>MODULE: sklearn is used, l-bfgs optimization.')
                            if VERBOSE: print(mt.kernel_, mt.log_marginal_likelihood(mt.kernel_.theta))
                        else:
                            print('>>>>>>MODULE: GPy is used, l-bfgs optimization.')
                            mt.optimize(optimizer='bfgs', gtol = 1e-100, messages=VERBOSE, max_iters=int(10000*bounds.shape[0]))
                            mt.optimize_restarts(num_restarts=int(10*bounds.shape[0]),robust=True,verbose=VERBOSE)
                    elif self.METHOD == 'BAYESIAN':
                        if self.USE_SKLEARN: sys.exit('FUTURE: Currently we cannot fit with Bayesian method using sklearn, we have GPy only.')
                        ##############################
                        #Fully Bayesian GP surrogate #
                        ##############################
                        #Prior on the "hyper-parameters" for the GP surrogate model.
                        print('>>>>>>METHOD: Fully Bayesian approach, component '+str(c)+'/'+str(len(component_label)-1))
                        print('>>>>>>SAMPLE: component sample size =',len(c_idx[0]) )
                        mt.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        mt.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        #HMC sampling, fully Bayesian approach to estimate the kernel parameters.
                        hmc = GPy.inference.mcmc.HMC(mt,stepsize=0.1)
                        s = hmc.sample(num_samples=self.N_BURNIN) # Burnin
                        s = hmc.sample(num_samples=self.N_MCMCSAMPLES)
                        MCMC_samples = s[self.N_INFERENCE:] # cut out the burn-in period
                        # Set the model parameters as the posterior mean
                        #mt.kern.variance[:]    = MCMC_samples[:,0].mean()
                        #mt.kern.lengthscale[:] = MCMC_samples[:,1].mean()
                    #######################################
                    # Optimization module(each component) #
                    #######################################
                    ##################################################
                    #    Maximization of the acquisition function    #
                    #To maximize my_obj, we can simply minimize this my_obj function.
                    def my_obj(X):
                        return acq_fun(X=X,surrogate=mt,X_sample=Xt,Y_sample=Yt,correct_label=c,classify=clf_XY,USE_SKLEARN=self.USE_SKLEARN,VERBOSE=VERBOSE,boundary_penalty=self.boundary_penalty) 

                    x0 = np.random.uniform(bounds[:,0].T,bounds[:,1].T)
                    ret = minimize(my_obj, x0, method='trust-constr',  jac="2-point", hess=SR1(),
                                #constraints=[],
                                constraints=[linear_constraint, nonlinear_constraint],
                                options={'verbose': 1}, bounds=bounds_constraint)

                    #ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=123)
                    #dual annealing works for dim=1
                    #                                                #
                    ##################################################
                    
                    #ret = minimize(fun=func, x0=, bounds=list(zip(lw, up)), method='L-BFGS-B')
                    print('>>>>Maximal acquisition function = ',-ret.fun,' attained at ',ret.x,' for component ',c)
                    object_model_list[int(c)]=mt
                    if -ret.fun>opt_acq:      
                        opt_x = ret.x
                        opt_acq = -ret.fun
                X_next = opt_x
            else:
                print('>>>>Find next sample: random search.')
                def my_obj(X):
                        return 0
                #Optimize this my_obj using some optimization method, this makes sure that the random choice lies within the constrained domain.
                ##################################################
                #    Maximization of the acquisition function    #
                func = my_obj
                x0 = np.random.uniform(bounds[:,0].T,bounds[:,1].T)
                ret = minimize(my_obj, x0, method='trust-constr',  jac="2-point", hess=SR1(),
                            #constraints=[],
                            constraints=[linear_constraint, nonlinear_constraint],
                            options={'verbose': 1}, bounds=bounds_constraint)

                #ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=123)
                #dual annealing works for dim=1
                #                                                #
                ##################################################
                X_next = ret.x
                if ~self.REPEAT_SAMPLE:
                        REPEAT_FLAG = True
                        while REPEAT_FLAG:
                                REPEAT_FLAG = False
                                for j in range(X_sample.shape[1]):
                                        #print(X_sample[j,:])
                                        if (X_next == X_sample[j,:]).all():
                                                REPEAT_FLAG = True
                                if REPEAT_FLAG:
                                        ret = minimize(my_obj, x0, method='trust-constr',  jac="2-point", hess=SR1(),
                                                #constraints=[],
                                                constraints=[linear_constraint, nonlinear_constraint],
                                                options={'verbose': 1}, bounds=bounds_constraint)
                                        X_next = ret.x
            X_next = X_next.reshape(1,-1)
            Y_next = self.f_truth(X_next)
            print('----------')
            print('>>Next sample input is chosen to be: ',X_next)
            print('>>Next sample response is chosen to be: ',Y_next.ravel())
            #Update X and Y from this step.
            X_sample = np.vstack((X_sample,X_next))
            Y_sample = np.vstack((Y_sample,self.censor_function(Y_next) ))
        ########################################
        #        Stop sequential samples       #
        ########################################

        sampleendingtime = datetime.now()
        # dd/mm/YY H:M:S
        print("Sample start date and time =", samplestartingtime)
        print("Sample end date and time =", sampleendingtime)
        if self.NO_CLUSTER==True:
            FILE_NAME = self.EXAMPLE_NAME+'_local_GP('+rdstr+')'
        else:
            FILE_NAME = self.EXAMPLE_NAME+'_local_cGP_k='+str(self.N_COMPONENTS)+'('+rdstr+')' 
        XY_sample_final = np.concatenate((self.X_AMPLIFY*(X_sample-self.X_TRANSLATE),self.Y_AMPLIFY*(Y_sample-self.Y_TRANSLATE)),axis=1)
        XY_label_final  = dgm_XY.fit_predict(XY_sample_final)
        XY_label_final  = XY_label_final.reshape(-1,1)
        #print(XY_label_final.shape)
        pre_var_names = ['X%d' % (i+1) for i in range(0, X_sample.shape[1], 1)]
        pre_var_names.insert(0,'Y')
        pre_var_names.append('label')
        var_names = ','.join(pre_var_names)
        #print(var_names)
        np.savetxt(FILE_NAME+'.txt', np.hstack((Y_sample,X_sample,XY_label_final)), delimiter =',',
                header=var_names,comments='',
                fmt=','.join(['%1.12f']*(1+X_sample.shape[1])+['%i']))  

        sample_max_x = X_sample[np.argmax(Y_sample),:] 
        sample_max_f = np.round( Y_sample[np.argmax(Y_sample),:],12)
        sample_min_x = X_sample[np.argmin(Y_sample),:] 
        sample_min_f = np.round( Y_sample[np.argmin(Y_sample),:],12)
        ########################################
        #        Write a fitting log file      #
        ########################################
        if True:
            original_stdout = sys.stdout # Save a reference to the original standard output
            with open(FILE_NAME+'.log', 'w') as f:
                sys.stdout = f # Change the standard output to the file we created.
                #print('This message will be written to a file.')
                print("Example: ",self.EXAMPLE_NAME,file=f)
                print("Sample start date and time = ", samplestartingtime)
                print("Sample end date and time = ", sampleendingtime)
                print("Python version: ", sys.version)
                #print("Filename of the script: ", sys.argv[0])
                # print("Commandline arguments: ",sys.argv)
                print("Random seed: ",self.RND_SEED)
                print('Random stamp: ',rdstr)
                print('GPy version: ', GPy.__version__)
                print('sklearn version: ', sklearn.__version__)
                print('Number of pilot samples: ',self.N_PILOT)
                print('Number of sequential samples: ',self.N_SEQUENTIAL)
                print('Surrogate fitting method: ',self.METHOD)
                if self.METHOD=="BAYESIAN":
                    print('MCMC>Burn-in steps: ',self.N_BURNIN)
                    print('MCMC>Sampling steps: ',self.N_MCMCSAMPLES)
                    print('MCMC>Inference sample length: ',self.N_INFERENCE)
                print('Surrogate> Are we using sklearn for GPR?: ',self.USE_SKLEARN)
                print('Surrogate> kernel type: ',KERNEL_TEMPLATE)
                if self.USE_SKLEARN:
                    print('Surrogate>sklearn>jittering: ',self.ALPHA_SKLEARN)
                    print('Surrogate>sklearn>normalizer; ',self.SKLEARN_normalizer)
                else:
                    print('Surrogate>GPy>normalizer; ',self.GPy_normalizer)
                print('Surrogate> Fit a simple GP?(no cluster): ',self.NO_CLUSTER)
                print('Cluster-classify> Response amplifier when clustering: ',self.Y_AMPLIFY)
                print('Cluster> DGM Maximal number of components/clusters: ',self.N_COMPONENTS)
                print('Classify> k in k-nearest neighbor classifier',self.N_NEIGHBORS)
                print('Exploration rate: ',self.EXPLORATION_RATE)
                print('Exploration> Do we allow repeat samples in random searching?',self.REPEAT_SAMPLE)
                print('domain bounds: ',bounds)
                print('sample minimum, f_min=',sample_min_f,' at ',sample_min_x)
                print('sample maximum, f_max=',sample_max_f,' at ',sample_max_x)
                print('>>Cluster X_AMPLIFY=',self.X_AMPLIFY)
                print('>>Cluster X_TRANSLATE=',self.X_TRANSLATE)
                print('>>Cluster Y_AMPLIFY=',self.Y_AMPLIFY)
                print('>>Cluster Y_TRANSLATE=',self.Y_TRANSLATE)
            sys.stdout = original_stdout # Reset the standard output to its original value

        #%debug
        import os
        print('Logs of run with stamp: ',rdstr,', is saved at',os.getcwd())


        print('sample minimum, f_min=',sample_min_f,' at ',sample_min_x)
        print('sample maximum, f_max=',sample_max_f,' at ',sample_max_x)
        print('>>Cluster X_AMPLIFY=',self.X_AMPLIFY)
        print('>>Cluster X_TRANSLATE=',self.X_TRANSLATE)
        print('>>Cluster Y_AMPLIFY=',self.Y_AMPLIFY)
        print('>>Cluster Y_TRANSLATE=',self.Y_TRANSLATE)


        print('Total sample size:',X_sample.shape[0])
        #print(np.concatenate((self.X_AMPLIFY*(X_sample-self.X_TRANSLATE),self.Y_AMPLIFY*(Y_sample-self.Y_TRANSLATE).reshape(-1,1)),axis=1))
        #print(np.concatenate((X_sample,self.Y_AMPLIFY*(Y_sample-0.).reshape(-1,1)),axis=1))
        #print('Run stamp:',rdstr)
        print('Run seed',self.RND_SEED)
        #print(sample_max_f)
        #print(sample_max_x)
        final_N_COMPONENTS = len(np.unique(np.array(cluster_label)))
        print('Final number of components:',final_N_COMPONENTS)
        ########################################
        #        Save the fitted model(s)      #
        ########################################
        #Dump the fitted model if we provide a final parameter as the object name.
        #import pickle as pkl
        import dill as pkl
        pkl_dict = {'Cluster':dgm_XY,'Classify':clf_XY,'GPR_list':object_model_list,'f_truth':self.f_truth,'get_bounds':self.get_bounds}
        # if len(sys.argv)>=12:
        #     OBJ_NAME = str(sys.argv[11]) + '.pkl'
        #     print('Ready to dump our cGP model...',OBJ_NAME)
        #     print(pkl_dict)  
        #     with open(OBJ_NAME, "wb+") as file_model: 
        #         pkl.dump(pkl_dict, file_model)

