#!/usr/bin/env python
# coding: utf-8
########################################
#     clustered Gaussian surrogate     #
########################################
#Author: Hengrui Luo
#hrluo@lbl.gov
#Last update: 2021-Apr-28
########################################
#          System information          #
########################################
#Print the python version and the name/input arguments
print('usage: python cGP_constrained.py RND_SEED(random seed) N_SEQUENTIAL(sequential sample size) EXPLORATION_RATE(exploration rate) NO_CLUSTER (1=simple GP;0=cGP) N_COMPONENTS(maximal # of components in cGP, OR name of cluster definition) N_NEIGHBORS(# of neighbors in k-NN classification, OR name of classfier definition) N_PILOT(pilot sample size, or passing a pilot sample file name) N_PROC(number of processors you want to use in fitting) F_TRUTH_PY(location of f_truth.py, no extension needed) OUTPUT_NAME(overriding filename, no extension needed.) ACQUISITION(name of acquistion function.flexible) OBJ_NAME(output model name, saving the surrogate model)')
print('example1: python cGP_parallel.py 123 30 1.0 0 4 3 1 10')
print("example2: python cGP_parallel.py 123 60 0.5 0 2 3 samples_sobol_10.txt 2 'f_truth1' 'test1'")
print("example3: python cGP_parallel.py 123 40 0.8 0 'cluster' 'classify' samples_random_10.txt 4 'f_truth' 'acquisition' 'test1' 'model1'")
#Clean up
import sys
sys.modules[__name__].__dict__.clear()
import sys
#Warnings supression
import warnings
warnings.filterwarnings('ignore')
print("Python version: ", sys.version)
print(sys.argv)

#Multi-processing
import multiprocessing
 
#Print the numpy version and set the random seed
import numpy as np
print('numpy version: ', np.__version__)
from numpy import int64	
from numpy import int	
from numpy import float	
from numpy import bool
RND_SEED = int(sys.argv[1])
np.random.seed(RND_SEED)
print('Seed=',RND_SEED)

#Get a random string stamp for this specific run, used for the filename of image export.
import random
import string
def get_random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
rdstr=get_random_string(8)
print('random stamp for this run:',rdstr)

#Print the GPy version
import GPy
print('GPy version: ', GPy.__version__)

#Print the sklearn version
import sklearn
print('sklearn version: ', sklearn.__version__)
from sklearn.gaussian_process import GaussianProcessRegressor

import importlib
#n_proc_value = input("Please enter the number of processors you want to use in parallel cGP component fitting:\n")
#print(f'You entered {n_proc_value}.\n There are ',multiprocessing.cpu_count(),' processors available.\n')
if len(sys.argv) >= 9:
	n_proc_value = int(sys.argv[8])
else:
	n_proc_value = 1
if int(n_proc_value)<=multiprocessing.cpu_count():
	n_proc = int(n_proc_value);
else:
	n_proc = 1;
	print('WARNING: Invalid number of processors, only 1 processor would be used.')
	
if len(sys.argv) >=10:
	task_name = str(sys.argv[9])
else:
	task_name = 'f_truth'

from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
#######################################
#         Model specification          #
########################################
#How many pilot and sequential samples do we allow to get?
#N_PILOT is the pilot samples we start with, usually a small number would do.
#N_SEQUENTIAL is the number of sequential (noisy) samples we should draw from the black-box function.
if sys.argv[7].isdigit():
	N_PILOT = int(sys.argv[7])
N_SEQUENTIAL =  int(sys.argv[2])
#Which method should we use for the Bayesian optimization scheme?
#'FREQUENTIST' method means that the (hyper-)parameters are estimated by using some frequestist optimization like lbfgs.
#'BAYESIAN' method means that the paramteres are estimated by putting a prior(Gamma)-posterior mechnism, the estimated value would be posterior mean.
METHOD  = 'FREQUENTIST'
#Following 3 parameters are only for HMC Bayesian sampling, you have to choose METHOD  = 'BAYESIAN' to use these parameters.
N_BURNIN = 500
N_MCMCSAMPLES = 500
N_INFERENCE = 300
#Exploration rate is the probability (between 0 and 1) of following the next step produced by acquisition function.
EXPLORATION_RATE = float(sys.argv[3])
#Do you want a cluster GP? If NO_CLUSTER = True, a simple GP will be used.
NO_CLUSTER = bool(int(sys.argv[4]))
print('NO_CLUSTER:',NO_CLUSTER,sys.argv[4])
#Do you want to amplify the weight/role of response X when doing clustering?
X_AMPLIFY = 1.
#Do you want to subtract an amount from the response X when doing clustering?
X_TRANSLATE = []
#Do you want to amplify the weight/role of response Y when doing clustering?
Y_AMPLIFY = 1.
#Do you want to subtract an amount from the response Y when doing clustering?
Y_TRANSLATE = 0.
#What is the maximal number of cluster by your guess? This option will be used only if NO_CLUSTER=False.
if sys.argv[6].isdigit():
	N_NEIGHBORS = int(sys.argv[6])
	print('\n > Classify: KNeighbors:',N_NEIGHBORS,'-neighbors')
	clf_XY = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
else:
	CLASSIFY_FUN = str(sys.argv[6])
	print('\n > Classify: ',CLASSIFY_FUN)
	clf_mdl = importlib.import_module(CLASSIFY_FUN)
	clf_mdl_names = [x for x in clf_mdl.__dict__ if not x.startswith("_")]
	globals().update({k: getattr(clf_mdl, k) for k in clf_mdl_names})
	clf_XY = f_Classify()
#When deciding cluster components, how many neighbors shall we look into and get their votes? This option will be used only if NO_CLUSTER=False.
if sys.argv[5].isdigit():
	N_COMPONENTS = int(sys.argv[5])
	print('\n > Cluster: BayesianGaussianMixture:',N_COMPONENTS,' components')
	dgm_XY = BayesianGaussianMixture(
                    #weight_concentration_prior_type="dirichlet_distribution",
                    weight_concentration_prior_type="dirichlet_process",
                    n_components=N_COMPONENTS,#pick a big number, DGM will automatically adjust
                    random_state=0)
	#dgm_XY = KMeans(n_clusters=N_COMPONENTS, random_state=0))
else:
	CLUSTER_FUN = str(sys.argv[5])
	print('\n > Cluster: ',CLUSTER_FUN)
	dgm_mdl = importlib.import_module(CLUSTER_FUN)
	dgm_mdl_names = [x for x in dgm_mdl.__dict__ if not x.startswith("_")]
	globals().update({k: getattr(dgm_mdl, k) for k in dgm_mdl_names})
	dgm_XY = f_Cluster(RND_SEED)
#This is a GPy parameter, whether you want to normalize the response before/after fitting. Don't change unless necessary.
GPy_normalizer = True
#Whether we should sample repetitive locations in the sequential sampling procedure.
#If True, we would keep identical sequential samples no matter what. (Preferred if we believe a lot of noise)
#If False, we would re-sample when we run into identical sequential samples. (Default)
#In a acquisition maximization step, this is achieved by setting the acquisition function at repetitive samples to -Inf
#In a random search step, this is achieved by repeat the random selection until we got a new location.
REPEAT_SAMPLE = False

USE_SKLEARN = True
ALPHA_SKLEARN = 1e-5
#Value added to the diagonal of the kernel matrix during fitting. 
SKLEARN_normalizer = True
########################################
#     Import model specification       #
########################################
mdl = importlib.import_module(task_name)
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    names = [x for x in mdl.__dict__ if not x.startswith("_")]
globals().update({k: getattr(mdl, k) for k in names})
bounds = get_bounds(1)	
print('bounds',bounds.shape)
inp_dim=bounds.shape[0]
########################################
#     Import acquisition function      #
########################################
if len(sys.argv)>=10:
    acquisition_name=str(sys.argv[9])
else:
    acquisition_name='expected_improvement'
print('>>>Acquisition function: ',acquisition_name)
acq = importlib.import_module('acquisition')
if "__all__" in acq.__dict__:
    names = acq.__dict__["__all__"]
else:
    names = [x for x in acq.__dict__ if not x.startswith("_")]
globals().update({k: getattr(acq, k) for k in names})
acq_fun = getattr(acq,acquisition_name)
########################################
#         Kernel specification         #
########################################
#Which kernel you want to use for your model? Such a kernel must be implemented as a GPy/sklearn kernel class.
if USE_SKLEARN==True:
    from sklearn.gaussian_process import *
    KERNEL_TEMPLATE = sklearn.gaussian_process.kernels.Matern(length_scale=np.ones(inp_dim,), length_scale_bounds=(1e-05, 100000.0), nu=3/2) + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05, 100000.0))
    #KERNEL_TEMPLATE = sklearn.gaussian_process.kernels.Matern(length_scale=np.ones(inp_dim,), length_scale_bounds=(1e-05, 100000.0), nu=1/2)
else:
    KERNEL_TEMPLATE = GPy.kern.Matern32(input_dim=inp_dim, variance=1., lengthscale=1.) + GPy.kern.White(input_dim=inp_dim)
    #KERNEL_TEMPLATE = GPy.kern.Exponential(input_dim=inp_dim, variance=1., lengthscale=1.)
#Do you want to penalize boundary sample points? If so, how?

from datetime import datetime
# datetime object containing current date and time
samplestartingtime = datetime.now()
########################################
#          Draw pilot samples          #
########################################
#This cell only provides a pilot sample.
#Prepare pilot samples (X,Y)
print('Example : ',EXAMPLE_NAME)
if not sys.argv[7].isdigit():
	print('\n>>>>>>>>>>Load pilot samples from external file: ',sys.argv[7],'...<<<<<<<<<<\n')
	X_sample = np.loadtxt( sys.argv[7] )
	N_PILOT = X_sample.shape[0]
else:
	print('\n>>>>>>>>>>Sampling ',N_PILOT,' pilot samples...<<<<<<<<<<\n')
	X_sample = np.zeros((N_PILOT,bounds.shape[0]))
	for j in range(bounds.shape[0]):
    		X_sample[:,j] = np.random.uniform(bounds[j,0],bounds[j,1],size=(N_PILOT,1)).ravel()
	
Y_sample = np.zeros((N_PILOT,1))
Y_sample = np.zeros((N_PILOT,1))
for k in range(N_PILOT):
    Y_sample[k,0] = f_truth(X_sample[k,:].reshape(1,-1))
    Y_sample[k,0] = censor_function(Y_sample[k,0])
#print('Pilot X',X_sample)
#print('Pilot Y',Y_sample)


#The cGP procedure consists of following steps
#Step 1. For observations, we can do a (unsupervised) (X,Y)-clustering and label them, different components are generated.
#Step 2. For predictive locations, we can do a (supervised) k-nearest neighbor classification, and predict at each location based on which component it belongs to. 
#Step 3. We compute the acquisition function and then proceed to the next sample, after adding the new sample we repeat Step 1 and 2.

#Prepare an up-to-date X_TRANSLATE, as the empirical mean of the X_sample
if len(X_TRANSLATE)>0:
    X_TRANSLATE = np.mean(X_sample,axis=0)
else:
    X_TRANSLATE = np.mean(X_sample,axis=0)*0
#Prepare an up-to-date Y_TRANSLATE, as the empirical mean of the Y_sample
if Y_TRANSLATE != 0:
    Y_TRANSLATE = np.mean(Y_sample)
#print(Y_sample - Y_TRANSLATE)
print(np.concatenate((X_sample,Y_AMPLIFY*(Y_sample-Y_TRANSLATE)),axis=1))
#Prepare initial clusters, with XY-joint.
XY_sample = np.concatenate((X_AMPLIFY*(X_sample-X_TRANSLATE),Y_AMPLIFY*(Y_sample-Y_TRANSLATE)),axis=1)
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
from scipy import stats
from scipy.optimize import SR1
from scipy.optimize import minimize
VERBOSE = False
#Prepare sequential samples (X,Y)
print('\n>>>>>>>>>>Sampling ',N_SEQUENTIAL,' sequential samples...<<<<<<<<<<\n')
X_sample = X_sample_XY
Y_sample = Y_sample_XY
cluster_label = XY_label

def get_KER():
    return KERNEL_TEMPLATE
#This recode function will turn the labels into increasing order,e.g. [1, 1, 3, 3, 0] ==> [0, 0, 1, 1, 2].
def recode(label):
    level = np.unique(np.array(label))
    ck = 0
    for j in level:
        label[label==j]=ck
        ck=ck+1
    return label
    
#Main loop that guides us in sampling sequential samples
component_label = np.unique(np.array(cluster_label))

for it in range(N_SEQUENTIAL):
    print('\n>>>>>>>>>> ***** STEP ',it+1,'/',N_SEQUENTIAL,'***** <<<<<<<<<<')
    print('\n>>>>>>>>>> +++++ N_PROC ',n_proc,' +++++ <<<<<<<<<<')
    #Step 1. For observations, we can do a (unsupervised) (X,Y)-clustering and label them, different components are generated.
    #Create the (X,Y) joint sample to conduct (unsupervised clustering)
    #if len(X_TRANSLATE)>0:
    #    X_TRANSLATE = np.mean(X_sample,axis=0)
    #if Y_TRANSLATE != 0:
    #    Y_TRANSLATE = np.mean(Y_sample)
    #The cluster must be based on adjusted response value Y.
    XY_sample        = np.concatenate((X_AMPLIFY*(X_sample-X_TRANSLATE),Y_AMPLIFY*(Y_sample-Y_TRANSLATE).reshape(-1,1)),axis=1)
    if NO_CLUSTER:
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
    cluster_label = recode(cluster_label)
    if VERBOSE: print('All labels are recoded: ',cluster_label)
    
    #Step 2. For predictive locations, we can do a (supervised) k-nearest neighbor classification, and predict at each location based on which component it belongs to. 
    #Refresh the Classifier
    if sys.argv[6].isdigit():
        clf_XY = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    else:
        clf_XY = f_Classify()
    clf_XY.fit(X_sample,cluster_label)
        
    #Step 3. We either randomly search one location or compute the acquisition function and then proceed to the next sample, after adding the new sample we repeat Step 1 and 2.
    coin = np.random.uniform(0,1,1)
    if coin<EXPLORATION_RATE:
        component_label = np.unique(np.array(cluster_label))
        if not NO_CLUSTER:
    	    print('>>CLUSTERED, a cGP surrogate.',len(component_label),' components in surrogate model.');    
        print('>>>>Find next sample: acquisition proposal.')
	##############################
	#Multi-processing mechanism  #
	##############################
        #
        global component_c;
        def component_c(c):
            #Assign the corresponding X_sample and Y_sample values to the cluster coded by c. Return the maximizer of acquisition function based on the samples from this component.  
            c_idx = np.where(cluster_label == int(c))
            if VERBOSE: 
                print('>>>>Fitting component ',c,'/',len(component_label)-1,' total components')
                print(c_idx)
            Xt = X_sample[c_idx].ravel().reshape(-1,X_sample.shape[1])
            Yt = Y_sample[c_idx].ravel().reshape(-1,1)
            #Fit the model with normalization
            if USE_SKLEARN==True:
                mt = GaussianProcessRegressor(kernel=get_KER(), random_state=0, normalize_y=SKLEARN_normalizer,alpha=ALPHA_SKLEARN,  
                                              optimizer='fmin_l_bfgs_b', n_restarts_optimizer=int(10*bounds.shape[0]))
            else:
                mt = GPy.models.GPRegression(Xt, Yt, kernel=get_KER(), normalizer=GPy_normalizer)
            ###
            if METHOD == 'FREQUENTIST':
                ##############################
                #Frequentist MLE GP surrogate#
                ##############################
                #print('>>>>>>METHOD: frequentist MLE approach, component '+str(c)+'/'+str(len(component_label)-1))
                #print('>>>>>>SAMPLE: component sample size =',len(c_idx[0]) )
                if USE_SKLEARN==True:
                    mt.fit(Xt, Yt)
                    #No need to do more for sklearn GP
                    #print('>>>>>>MODULE: sklearn is used, l-bfgs optimization.')
                    #if VERBOSE: print(mt.kernel_, mt.log_marginal_likelihood(mt.kernel_.theta))
                else:
                    #print('>>>>>>MODULE: GPy is used, l-bfgs optimization.')
                    mt.optimize(optimizer='bfgs', gtol = 1e-100, messages=VERBOSE, max_iters=int(10000*bounds.shape[0]))
                    mt.optimize_restarts(num_restarts=int(10*bounds.shape[0]),robust=True,verbose=VERBOSE)
            elif METHOD == 'BAYESIAN':
                if USE_SKLEARN: sys.exit('FUTURE: Currently we cannot fit with Bayesian method using sklearn, we have GPy only.')
                ##############################
                #Fully Bayesian GP surrogate #
                ##############################
                #Prior on the "hyper-parameters" for the GP surrogate model.
                #print('>>>>>>METHOD: Fully Bayesian approach, component '+str(c)+'/'+str(len(component_label)-1))
                #print('>>>>>>SAMPLE: component sample size =',len(c_idx[0]) )
                mt.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                mt.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                #HMC sampling, fully Bayesian approach to estimate the kernel parameters.
                hmc = GPy.inference.mcmc.HMC(mt,stepsize=0.1)
                s = hmc.sample(num_samples=N_BURNIN) # Burnin
                s = hmc.sample(num_samples=N_MCMCSAMPLES)
                MCMC_samples = s[N_INFERENCE:] # cut out the burn-in period
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
                return acq_fun(X=X,surrogate=mt,X_sample=Xt,Y_sample=Yt,correct_label=c,classify=clf_XY,USE_SKLEARN=USE_SKLEARN,VERBOSE=VERBOSE,boundary_penalty=boundary_penalty) 

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
            #print('>>>>Maximal acquisition function = ',-ret.fun,' attained at ',ret.x,' for component ',c)
            #X_next = ret.x
            return ret.x,-ret.fun,mt
            
        # create a "pool" of 4 processes to do the calculations
        pool = multiprocessing.Pool(processes=n_proc)
        input_values = [c for c in component_label ]
        #print(input_values)
        list2 = pool.map(component_c, input_values)
        pool.close()
        if VERBOSE: print('For each of ',len(component_label),' components, we yield:',np.asarray(list2))
        #for c in component_label:
        object_model_list = [ r2[2] for r2 in list2 ]
        list2_fun = [ r2[1] for r2 in list2 ]
        opt_ind = np.argmax(np.asarray(list2_fun))
        opt_x = list2[opt_ind]
        X_next = opt_x[0]
        
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
        if ~REPEAT_SAMPLE:
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
    Y_next = f_truth(X_next)
    print('----------')
    print('>>Next sample input is chosen to be: ',X_next)
    print('>>Next sample response is chosen to be: ',Y_next.ravel())
    #Update X and Y from this step.
    X_sample = np.vstack((X_sample,X_next))
    Y_sample = np.vstack((Y_sample,censor_function(Y_next) ))
########################################
#        Stop sequential samples       #
########################################

sampleendingtime = datetime.now()
# dd/mm/YY H:M:S
print("Sample start date and time =", samplestartingtime)
print("Sample end date and time =", sampleendingtime)
if len(sys.argv)>=12:
    FILE_NAME = str(sys.argv[11])
else:
    if NO_CLUSTER==True:
        FILE_NAME = EXAMPLE_NAME+'_local_GP('+rdstr+')'
    else:
        if sys.argv[5].isdigit():
            FILE_NAME = EXAMPLE_NAME+'_local_cGP_k='+str(N_COMPONENTS)+'('+rdstr+')'
        else:
            FILE_NAME = EXAMPLE_NAME+'_local_cGP_['+str(sys.argv[5])+']('+rdstr+')' 
XY_sample_final = np.concatenate((X_AMPLIFY*(X_sample-X_TRANSLATE),Y_AMPLIFY*(Y_sample-Y_TRANSLATE)),axis=1)
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
        print("Example: ",EXAMPLE_NAME,file=f)
        print("Sample start date and time = ", samplestartingtime)
        print("Sample end date and time = ", sampleendingtime)
        print("Python version: ", sys.version)
        #print("Filename of the script: ", sys.argv[0])
        print("Commandline arguments: ",sys.argv)
        print("Random seed: ",RND_SEED)
        print('Random stamp: ',rdstr)
        print('GPy version: ', GPy.__version__)
        print('sklearn version: ', sklearn.__version__)
        print('Number of pilot samples: ',N_PILOT)
        if not sys.argv[7].isdigit():
	        print('pilot sample files: ',sys.argv[7])
        print('Number of sequential samples: ',N_SEQUENTIAL)
        print('Surrogate fitting method: ',METHOD)
        if METHOD=="BAYESIAN":
            print('MCMC>Burn-in steps: ',N_BURNIN)
            print('MCMC>Sampling steps: ',N_MCMCSAMPLES)
            print('MCMC>Inference sample length: ',N_INFERENCE)
        print('Surrogate> Are we using sklearn for GPR?: ',USE_SKLEARN)
        print('Surrogate> kernel type: ',get_KER())
        if USE_SKLEARN:
            print('Surrogate>sklearn>jittering: ',ALPHA_SKLEARN)
            print('Surrogate>sklearn>normalizer; ',SKLEARN_normalizer)
        else:
            print('Surrogate>GPy>normalizer; ',GPy_normalizer)
        print('Surrogate> Fit a simple GP?(no cluster): ',NO_CLUSTER)
        print('Cluster-classify> Response amplifier when clustering: ',Y_AMPLIFY)
        if not sys.argv[5].isdigit():
            print('Cluster> ',sys.argv[5])
        else:
            print('Cluster> DGM Maximal number of components/clusters: ',N_COMPONENTS)
        if not sys.argv[6].isdigit():
            print('Classify> ',sys.argv[6])
        else:
            print('Classify> k in k-nearest neighbor classifier',N_NEIGHBORS)
        print('Exploration rate: ',EXPLORATION_RATE)
        print('Exploration> Do we allow repeat samples in random searching?',REPEAT_SAMPLE)
        print('domain bounds: ',bounds)
        print('sample minimum, f_min=',sample_min_f,' at ',sample_min_x)
        print('sample maximum, f_max=',sample_max_f,' at ',sample_max_x)
        print('>>Cluster X_AMPLIFY=',X_AMPLIFY)
        print('>>Cluster X_TRANSLATE=',X_TRANSLATE)
        print('>>Cluster Y_AMPLIFY=',Y_AMPLIFY)
        print('>>Cluster Y_TRANSLATE=',Y_TRANSLATE)
    sys.stdout = original_stdout # Reset the standard output to its original value

#%debug
import os
print('Logs of run with stamp: ',rdstr,', is saved at',os.getcwd())


print('sample minimum, f_min=',sample_min_f,' at ',sample_min_x)
print('sample maximum, f_max=',sample_max_f,' at ',sample_max_x)
print('>>Cluster X_AMPLIFY=',X_AMPLIFY)
print('>>Cluster X_TRANSLATE=',X_TRANSLATE)
print('>>Cluster Y_AMPLIFY=',Y_AMPLIFY)
print('>>Cluster Y_TRANSLATE=',Y_TRANSLATE)


print('Total sample size:',X_sample.shape[0])
#print(np.concatenate((X_AMPLIFY*(X_sample-X_TRANSLATE),Y_AMPLIFY*(Y_sample-Y_TRANSLATE).reshape(-1,1)),axis=1))
#print(np.concatenate((X_sample,Y_AMPLIFY*(Y_sample-0.).reshape(-1,1)),axis=1))
#print('Run stamp:',rdstr)
print('Run seed',RND_SEED)
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
pkl_dict = {'Cluster':dgm_XY,'Classify':clf_XY,'GPR_list':object_model_list,'f_truth':f_truth,'get_bounds':get_bounds}
if len(sys.argv)>=13:
    OBJ_NAME = str(sys.argv[12]) + '.pkl'
    print('Ready to dump our cGP model...',OBJ_NAME)
    print(pkl_dict)  
    with open(OBJ_NAME, "wb+") as file_model: 
    	pkl.dump(pkl_dict, file_model)

