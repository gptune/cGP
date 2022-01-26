import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
def f_Cluster(RND_SEED):
    return KMeans(n_clusters=3, 
                    random_state=RND_SEED)
	#return BayesianGaussianMixture(
    #                #weight_concentration_prior_type="dirichlet_distribution",
    #                weight_concentration_prior_type="dirichlet_process",
    #                n_components=4,#pick a big number, DGM will automatically adjust
    #                random_state=RND_SEED)
	
#This file defines the constructor of Cluster
#The returning f_Cluster class object should have at least 1 methods
#f_Cluster.predict(Predictive samples)

