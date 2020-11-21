# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:35:29 2020

@author: haixiang_zhang
Generate two queues under resource constraints examples.
"""

import math
import numpy as np
from scipy import stats

def QueueModel(params):
    """
    Generate a queueing model with two queues
    """
    
    # Generate intensity function
    X = stats.uniform.rvs(0.75,1.25)
    Y = stats.uniform.rvs(0.75,1.25)
    Z = stats.uniform.rvs(-0.5,0.5)
    Gamma_1 = X + Z
    Gamma_2 = Y - Z
    lambda_1 = lambda t: Gamma_1 * ( 300 + 100 * math.sin(0.3*t) )
    lambda_2 = lambda t: Gamma_2 * ( 500 + 200 * math.sin(0.2*t) )
    
    
    
    
    
    
    
    
    return 0

def SingleQueue(num_server,intensity,params):
    """
     Compute the waiting time of a single queue
    """
    
    pass