import math
import datetime
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# System parameters
ACT365FIX = 365.0
zero      = 0.05

# Evaluation parameters
today                  = '2021/04/01'
application_start_date = '2021/06/01'
application_end_date   = '2021/09/01'

# Hull-White parameters
# Global
a     = 0.1
sigma = 0.005

def date_to_serial(date):
    # date is expected as string, 'YYYY/MM/DD'
    dt = datetime.datetime.strptime(date, '%Y/%m/%d')- datetime.datetime(1899, 12, 31)
    serial = dt.days + 1
    return serial

def DF(t,T):
    _t = date_to_serial(t)
    _T = date_to_serial(T)
    _tau = (_T-_t)/ACT365FIX
    
    return math.exp(-zero*_tau)

def FWD(s,t,T):
    _t = date_to_serial(t)
    _T = date_to_serial(T)
    _tau = (_T - _t) / ACT365FIX
    _fwd = (DF(s,t)/DF(s,T)-1.0)/_tau
    
    return _fwd

def B(s,t):
    s = date_to_serial(s)
    t = date_to_serial(t)
    tmp = (1-math.exp(-a*(t-s)))/a
    return tmp

def M(s,t,T):
    s = date_to_serial(s)
    t = date_to_serial(t)
    T = date_to_serial(T)
    tmp = (sigma**2)/(a**2)*(1-math.exp(-a*(t-s)))\
        -0.5*(sigma**2)/(a**2)*(math.exp(-a*(T-t))-math.exp(-a*(T-s+t-s)))
    return tmp

def V(s,t):
    s = date_to_serial(s)
    t = date_to_serial(t)
    tmp = 0.5*((sigma**2)/(a**3))*(2*a*(t-s)-3+4*math.exp(-a*(t-s))-math.exp(-2*a*(t-s)))
    return tmp

def ND_F(x):
    return norm.cdf(x, loc=0, scale=1)

def PD_F(x):
    return norm.pdf(x, loc=0, scale=1)

def _T(today, application_start_date, application_end_date):
    int_term = application_end_date-application_start_date
    pre_term = application_start_date-today
    
    tmp = 1/(2*a)*(((1-math.exp(-a*int_term))/a)**2) * (1-math.exp(-2*a*pre_term)) \
        + 1/(a**2)*(int_term+2/a*math.exp(-a*int_term)-1/(2*a)*math.exp(-2*a*int_term)-3/(2*a))
    return tmp

def caplet_compound(today, rate, strike, application_start_date, application_end_date):
    today                  = date_to_serial(today)
    application_start_date = date_to_serial(application_start_date)
    application_end_date   = date_to_serial(application_end_date)
    
    tau = (application_end_date - application_start_date) / ACT365FIX
    d_1 = (math.log((1.0+tau*rate)/(1.0+tau*strike))+(sigma**2)*0.5*_T(today, application_start_date, application_end_date))/(sigma*math.sqrt(_T(today, application_start_date, application_end_date)))
    d_2 = d_1 - sigma*math.sqrt(_T(today, application_start_date, application_end_date))
    
    return ((1.0+tau*rate)*ND_F(d_1)-(1.0+tau*strike)*ND_F(d_2))

def caplet_arithmetic_average(today, rate, strike, application_start_date, application_end_date):
    today                  = date_to_serial(today)
    application_start_date = date_to_serial(application_start_date)
    application_end_date   = date_to_serial(application_end_date)
    
    tau = (application_end_date - application_start_date) / ACT365FIX
    d   = (tau*rate-tau*strike)/(sigma*math.sqrt(_T(today, application_start_date, application_end_date)))
    
    return ((tau*rate-tau*strike)*ND_F(d)+sigma*math.sqrt(_T(today, application_start_date, application_end_date))*PD_F(d))

# main method start... 
mu = -B(application_start_date, application_end_date) * M(today, application_start_date, application_end_date) \
   -  V(application_start_date, application_end_date) + math.log(DF(today, application_start_date)/DF(today, application_end_date))\
   + 0.5*(V(today, application_end_date)-V(today, application_start_date))

variance = (B(application_start_date, application_end_date)**2)*(sigma**2)/(2*a)*(1-math.exp(-2*a*(date_to_serial(application_start_date)-date_to_serial(today))))+\
         + (sigma**2)/(a**2)*(date_to_serial(application_end_date) - date_to_serial(application_start_date) + (2/a)*math.exp(-a*(date_to_serial(application_end_date) - date_to_serial(application_start_date)))-1/(2*a)*math.exp(-2*a*(date_to_serial(application_end_date) - date_to_serial(application_start_date)))-3/(2*a))

aylst_lognormal = np.random.lognormal(mu, math.sqrt(variance), 1000000000)
#count, bins, ignored = plt.hist(aylst_lognormal, 1000, density=True, align='mid')
#np.savetxt('comp_out.csv',rv,delimiter=',')
#plt.show()

aylst_normal    =    np.random.normal(mu, math.sqrt(variance), 1000000000)
#count, bins, ignored = plt.hist(aylst_normal, 1000, density=True, align='mid')
#np.savetxt('comp_out.csv',rv,delimiter=',')
#plt.show()

strike_factor_list = [1.5,1.25,1.0,0.75,0.5]

# For compound
underlying = FWD(today, application_start_date, application_end_date)
for strike_factor in strike_factor_list:
    strike = underlying * strike_factor
    print("[underlying, strike] = ", str([underlying, strike]))
    print("Analytical solution for Comp.", str(caplet_compound(today, \
                                                    underlying, \
                                                    strike, \
                                                    application_start_date, \
                                                    application_end_date)))
   
    _tau = (date_to_serial(application_end_date) - date_to_serial(application_start_date))/ACT365FIX
    adj_strike = 1 + _tau*strike
    payoff_comp = aylst_lognormal - adj_strike
    payoff_comp[payoff_comp < 0.0] = 0.0
    #calplet_detail = pd.DataFrame(np.array([aylst, aylst - adj_strike, payoff]).T, \
                                    #columns=['original_value','original_value - adjstrike','payoff'])
    
    print("MonteCarlo solution for Comp.", str(payoff_comp.mean()))
    #print("MonteCarlo solution", str(calplet_detail['payoff'].mean()))

del payoff_comp
gc.collect()

print()
print('-------------------')
print()

# for arithmetic average
tau = (date_to_serial(application_end_date) - date_to_serial(application_start_date)) / ACT365FIX
underlying = (-B(application_start_date, application_end_date) * M(today, application_start_date, application_end_date) \
   -  V(application_start_date, application_end_date) + math.log(DF(today, application_start_date)/DF(today, application_end_date))\
   + 0.5*(V(today, application_end_date)-V(today, application_start_date)))/tau
for strike_factor in strike_factor_list:
    strike = underlying * strike_factor
    print("[underlying, strike] = ", str([underlying, strike]))
    print("Analytical solution for AA.", str(caplet_arithmetic_average(today, \
                                                                       underlying, \
                                                                       strike, \
                                                                       application_start_date, \
                                                                       application_end_date)))
    
    _tau = (date_to_serial(application_end_date) - date_to_serial(application_start_date))/ACT365FIX
    adj_strike = _tau*strike
    payoff_arithmetic_ave = aylst_normal - adj_strike
    payoff_arithmetic_ave[payoff_arithmetic_ave < 0.0] = 0.0
    #calplet_detail = pd.DataFrame(np.array([payoff_arithmetic_ave, payoff_arithmetic_ave - adj_strike, payoff]).T, \
                                    #columns=['original_value','original_value - adjstrike','payoff'])
    
    print("MonteCarlo solution for AA.", str(payoff_arithmetic_ave.mean()))
    #print("MonteCarlo solution", str(calplet_detail['payoff'].mean()))

del payoff_arithmetic_ave
gc.collect()