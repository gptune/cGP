#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC  # Updated import statement
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter

# Define the truth function
def f_truth(X):
    x, y = X
    return (y > 0) * (1 / (1 + (x - .25)**2 + (y - .25)**2)) + (y <= 0) * (.25 / (1 + (x**2 + y**2)))

# Define the search space
space = [Real(-1, 1, name='x'), Real(-1, 1, name='y')]

# Wrapper function for skopt
@use_named_args(space)
def objective_skopt(**params):
    X = np.array([params['x'], params['y']])
    return -f_truth(X)  # Negating since skopt minimizes

# Wrapper function for SMAC
def objective_smac(cfg):
    cfg = cfg.get_dictionary()
    x, y = cfg["x"], cfg["y"]
    return -f_truth([x, y])  # Negating since SMAC minimizes

# Optimization method wrapper
def optimize(method, seeds, n_calls=100):
    
    for seed in seeds:
        results = []
        
        if method == 'smac':
            # SMAC Configuration
            cs = ConfigurationSpace()
            x = UniformFloatHyperparameter("x", -1, 1)
            y = UniformFloatHyperparameter("y", -1, 1)
            cs.add_hyperparameters([x, y])
            scenario = Scenario({"run_obj": "quality", "runcount-limit": n_calls, "cs": cs, "deterministic": "true", "output_dir": "smac_output"})
            smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=objective_smac)
            incumbent = smac.optimize()
            rh = smac.get_runhistory()
            for run_key in rh.data:
                run = rh.data[run_key]
                config = cs.sample_configuration().get_dictionary()  # Get configuration values
                results.append([-run.cost, config["x"], config["y"]])
        else:
            # Scikit-Optimize
            if method == 'gp':
                res = gp_minimize(objective_skopt, space, n_calls=n_calls, random_state=seed)
            elif method == 'forest':
                res = forest_minimize(objective_skopt, space, n_calls=n_calls, random_state=seed, base_estimator='RF')
            elif method == 'gbrt':
                res = gbrt_minimize(objective_skopt, space, n_calls=n_calls, random_state=seed)
            elif method == 'dummy':
                res = dummy_minimize(objective_skopt, space, n_calls=n_calls, random_state=seed)
            
            for x in res.x_iters:
                results.append([-f_truth(x), x[0], x[1]])
    
        df = pd.DataFrame(results, columns=['y', 'x1', 'x2'])
        df.to_csv(f'optimization_results_f_4_{method}_{seed}.csv', index=False)

# Seeds
seeds = range(100, 201)

# Run optimizations
methods = ['gp', 'forest', 'gbrt', 'dummy', 'smac']
for method in methods:
    optimize(method, seeds)


# In[ ]:




