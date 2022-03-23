#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black-Scholes-Merton Equation in Python 

Adapted, documented and extended from Dr. Yves J. Hilpisch 
"Derivatives Analytics in Python"

Created on Sat Dec 18 20:19:30 2021

@author: kjwallace
"""

import math 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['font.family'] = 'calibri'
from scipy.integrate import quad 

#
# Helper Functions 
#
#


'''Probability density function of a standard normal variable x'''
def dN(x):
    return math.exp(-0.5 * x ** 2)/math.sqrt(2 * math.pi)


'''CDF of a standard normal variable x'''
def N(d):
    return quad(lambda x: dN(x), -20, d, limit = 50)[0]

'''d1 Terms of the BSM model'''

def d1f(S_t, K, t, T, r, sigma):
    d1 = (math.log(S_t / K) + (r + 0.5 * sigma ** 2) *(T-t)) / (sigma * math.sqrt(T-t))
    return d1

#
# Valuation Functions 
#

def BSM_call_value(S_t, K, t, T, r, sigma):
    '''Parameters
    =============
    S_t : float 
        spot price at time t 
    K: float 
        strike price 
    t: float 
        current time 
    T: float
        date of maturity 
    r: float 
        the risk free interest rate 
    sigma: float 
        annualized volatility 
        
    Returns call_value: float, the BSM value of the call '''
    
    d1 = d1f(S_t,K,t,T,r,sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    call_value = S_t * N(d1) - math.exp(-r*(T-t))*K*N(d2)
    return call_value
        
        
'''Returns the put value of an option with the same parameters as the call function'''
def BSM_put_value(S_t, K, t, T, r, sigma):
    put_value = BSM_call_value(S_t, K, t, T, r, sigma) - S_t + math.exp(-r * (T-t)) * K
    return put_value     
        
