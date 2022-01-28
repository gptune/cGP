import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def f_Classify():
	return KNeighborsClassifier(n_neighbors=3)
	
#This file defines the constructor of Classifier
#The returning f_Classify class object should have at least 2 methods
#f_Classify.fit(Observed samples,Observed labels)
#f_Classify.predict(Predictive samples)

