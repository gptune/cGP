#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter

# Define the robot arm function
def robot_arm_function(theta, L):
    u = sum(L[i] * np.cos(sum(theta[:i + 1])) for i in range(4))
    v = sum(L[i] * np.sin(sum(theta[:i + 1])) for i in range(4))
    return np.sqrt(u**2 + v**2)

# Define the search space for theta and L
space = [Real(0, 2*np.pi, name=f'theta{i+1}') for i in range(4)] + [Real(0, 1, name=f'L{i+1}') for i in range(4)]

# Wrapper function for skopt
@use_named_args(space)
def objective_skopt(**params):
    theta = [params[f'theta{i+1}'] for i in range(4)]
    L = [params[f'L{i+1}'] for i in range(4)]
    return robot_arm_function(theta, L)

# Wrapper function for SMAC
def objective_smac(cfg):
    cfg = cfg.get_dictionary()
    theta = [cfg[f'theta{i+1}'] for i in range(4)]
    L = [cfg[f'L{i+1}'] for i in range(4)]
    return robot_arm_function(theta, L)

# Optimization method wrapper
def optimize(method, seeds, n_calls=100):
    
    for seed in seeds:
        results = []
        if method == 'smac':
            # SMAC Configuration
            cs = ConfigurationSpace()
            hyperparameters = [UniformFloatHyperparameter(f'theta{i+1}', 0, 2*np.pi) for i in range(4)] + \
                              [UniformFloatHyperparameter(f'L{i+1}', 0, 1) for i in range(4)]
            cs.add_hyperparameters(hyperparameters)
            scenario = Scenario({"run_obj": "quality", "runcount-limit": n_calls, "cs": cs, "deterministic": "true", "output_dir": "smac_output"})
            smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=objective_smac)
            incumbent = smac.optimize()
            rh = smac.get_runhistory()
            for run_key in rh.data:
                run = rh.data[run_key]
                config = cs.sample_configuration().get_dictionary()  # Get configuration values
                results.append([run.cost, *[config[f'theta{i+1}'] for i in range(4)], *[config[f'L{i+1}'] for i in range(4)]])
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
                theta = x[:4]
                L = x[4:]
                results.append([robot_arm_function(theta, L), *theta, *L])
    
        df = pd.DataFrame(results, columns=['y', 'theta1', 'theta2', 'theta3', 'theta4', 'L1', 'L2', 'L3', 'L4'])
        df.to_csv(f'optimization_results_robot_{method}_{seed}.csv', index=False)

# Seeds
seeds = range(0, 51)

# Run optimizations
methods = ['gp', 'forest', 'gbrt', 'dummy', 'smac']
for method in methods:
    optimize(method, seeds)


# In[ ]:

