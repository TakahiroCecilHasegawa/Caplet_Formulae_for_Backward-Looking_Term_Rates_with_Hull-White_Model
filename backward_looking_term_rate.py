import math
import datetime
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# System parameters
ACT365FIX = 365.0

# Market Assumption parameters
zero                      = 0.05
historical_overnight_rate = 0.05

# Evaluation parameters
today                  = '2021/05/01'
application_start_date = '2021/04/01'
application_end_date   = '2021/11/01'

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
    _t   = date_to_serial(t)
    _T   = date_to_serial(T)
    _tau = (_T-_t)/ACT365FIX
    _df  = math.exp(-zero*_tau)
    return _df

def FWD(s,t,T, comp_flag = True):
    _fwd = 0.0
    _s = date_to_serial(s)
    _t = date_to_serial(t)
    _T = date_to_serial(T)
    _tau = (_T - _t) / ACT365FIX
    if comp_flag: # Compounding
        if _s < _t:
            _fwd = (DF(s,t)/DF(s,T)-1.0)/_tau
        elif _t <= _s:
            _m = _s - _t
            _fixing_comp = 1.0
            for _k in range(_m):
                _oneday_tau = 1.0/ACT365FIX
                # TODO historical_overnight_rate shold be a list that stores historical overnight rate
                _fixing_comp = _fixing_comp * (1+_oneday_tau*historical_overnight_rate)

            _fwd = (_fixing_comp/DF(s,T)-1.0)/_tau

    else: # Arithmetic Average
        _fwd = mu(s, t, T)/_tau

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
    tmp = 0.0
    if today < application_start_date:
        int_term = application_end_date   - application_start_date
        pre_term = application_start_date - today
        tmp = 1/(2*a)*(((1-math.exp(-a*int_term))/a)**2) * (1-math.exp(-2*a*pre_term)) \
            + 1/(a**2)*(int_term+2/a*math.exp(-a*int_term)-1/(2*a)*math.exp(-2*a*int_term)-3/(2*a))
    
    elif application_start_date <= today:
        int_term = application_end_date - today
        tmp = 1/(a**2)*(int_term+2/a*math.exp(-a*int_term)-1/(2*a)*math.exp(-2*a*int_term)-3/(2*a))
    
    return tmp

def caplet_pricer(today, rate, strike, application_start_date, application_end_date, comp_flag = True):
    
    price = 0.0
    if comp_flag:
        price = _caplet_compound(today, rate, strike, application_start_date, application_end_date)
    
    elif not comp_flag:
        price = _caplet_arithmetic_average(today, rate, strike, application_start_date, application_end_date)
    
    return price

def _caplet_compound(today, rate, strike, application_start_date, application_end_date):
    today                  = date_to_serial(today)
    application_start_date = date_to_serial(application_start_date)
    application_end_date   = date_to_serial(application_end_date)
    
    tau = (application_end_date - application_start_date) / ACT365FIX
    d_1 = (math.log((1.0+tau*rate)/(1.0+tau*strike))+(sigma**2)*0.5*_T(today, application_start_date, application_end_date))/(sigma*math.sqrt(_T(today, application_start_date, application_end_date)))
    d_2 = d_1 - sigma*math.sqrt(_T(today, application_start_date, application_end_date))
    _price = ((1.0+tau*rate)*ND_F(d_1)-(1.0+tau*strike)*ND_F(d_2))

    return _price

def _caplet_arithmetic_average(today, rate, strike, application_start_date, application_end_date):
    today                  = date_to_serial(today)
    application_start_date = date_to_serial(application_start_date)
    application_end_date   = date_to_serial(application_end_date)
    
    tau = (application_end_date - application_start_date) / ACT365FIX
    d   = (tau*rate-tau*strike)/(sigma*math.sqrt(_T(today, application_start_date, application_end_date)))
    
    return ((tau*rate-tau*strike)*ND_F(d)+sigma*math.sqrt(_T(today, application_start_date, application_end_date))*PD_F(d))

def mu(today, application_start_date, application_end_date):
    _mu = 0.0

    if date_to_serial(today) < date_to_serial(application_start_date):
        _mu = -B(application_start_date, application_end_date) * M(today, application_start_date, application_end_date) \
            - V(application_start_date, application_end_date) + math.log(DF(today, application_start_date)/DF(today, application_end_date))\
            + 0.5*(V(today, application_end_date)-V(today, application_start_date))

    else: # date_to_serial(application_start_date) <= date_to_serial(today):
        _sum = 0.0
        _s = date_to_serial(today)
        _t = date_to_serial(application_start_date)
        _m = _s - _t
        for _k in range(_m):
            _oneday_tau = 1.0/ACT365FIX
            # TODO historical_overnight_rate shold be a list that stores historical overnight rate
            _sum = _sum + _oneday_tau * historical_overnight_rate
        _mu = _sum - 0.5*V(today, application_end_date) - math.log(DF(today, application_end_date))
    
    return _mu

def val(today, application_start_date, application_end_date):
    _val = 0.0

    if date_to_serial(today) < date_to_serial(application_start_date):
        _val = (B(application_start_date, application_end_date)**2)*(sigma**2)/(2*a)*(1-math.exp(-2*a*(date_to_serial(application_start_date)-date_to_serial(today))))+\
           (sigma**2)/(a**2)*(date_to_serial(application_end_date) - date_to_serial(application_start_date) + (2/a)*math.exp(-a*(date_to_serial(application_end_date) - date_to_serial(application_start_date)))-1/(2*a)*math.exp(-2*a*(date_to_serial(application_end_date) - date_to_serial(application_start_date)))-3/(2*a))

    else: # date_to_serial(application_start_date) <= date_to_serial(today):
        _val = (sigma**2)/(a**2)*(date_to_serial(application_end_date) - date_to_serial(today) + (2/a)*math.exp(-a*(date_to_serial(application_end_date) - date_to_serial(today)))-1/(2*a)*math.exp(-2*a*(date_to_serial(application_end_date) - date_to_serial(today)))-3/(2*a))   
    return _val



# main method start... 
mu_num  = mu(today, application_start_date, application_end_date)
val_num = val(today, application_start_date, application_end_date)
aylst_lognormal = np.random.lognormal(mu_num, math.sqrt(val_num), 100000000)
aylst_normal    = np.random.normal(mu_num, math.sqrt(val_num),    100000000)

strike_weight_list = [1.5,1.25,1.0,0.75,0.5]

# For compound
underlying = FWD(today, application_start_date, application_end_date, True)
for strike_weight in strike_weight_list:
    strike = underlying * strike_weight
    print("[underlying, strike] = ", str([underlying, strike]))
    print("Analytical solution for Comp.", str(caplet_pricer(today, \
                                                    underlying, \
                                                    strike, \
                                                    application_start_date, \
                                                    application_end_date, \
                                                    True)))

    _tau = (date_to_serial(application_end_date) - date_to_serial(application_start_date))/ACT365FIX
    adj_strike = 1 + _tau*strike
    payoff_comp = aylst_lognormal - adj_strike
    payoff_comp[payoff_comp < 0.0] = 0.0
    print("MonteCarlo solution for Comp.", str(payoff_comp.mean()))
    print()

print()
print('-------------------')
print()

# For Arithmetic Average
underlying = FWD(today, application_start_date, application_end_date, False)
for strike_weight in strike_weight_list:
    strike = underlying * strike_weight
    print("[underlying, strike] = ", str([underlying, strike]))
    print("Analytical solution for AA.", str(caplet_pricer(today, \
                                                    underlying, \
                                                    strike, \
                                                    application_start_date, \
                                                    application_end_date, \
                                                    False)))

    _tau = (date_to_serial(application_end_date) - date_to_serial(application_start_date))/ACT365FIX
    adj_strike = _tau*strike
    payoff_AA = aylst_normal - adj_strike
    payoff_AA[payoff_AA < 0.0] = 0.0
    print("MonteCarlo solution for AA.", str(payoff_AA.mean()))
    print()

print()
print('-------------------')
print()
