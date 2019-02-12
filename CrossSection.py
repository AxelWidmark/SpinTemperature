#!/Users/axelwidmark/anaconda/bin/python
import math
import numpy as np
import matplotlib.pyplot as plt
pi=math.pi
from scipy.integrate import quad

# assume mass mH = 1 GeV
# hyperfine energy in GeV (0.068 K)
E_hf = 5.87433e-15
# Boltzmann constant in GeV/K
k_B = 8.61733e-14
# Speed of light in units km/s
c = 2.998e5
# Hydrogen mass in GeV
H_mass = 0.9388


def form_factor(v_in,mX,interaction): # takes v_in in units km/s
    mu = mX/(H_mass+mX)/c**2.
    if interaction=='elastic':
        res = 1.
    elif interaction=='deexcitation':
        res = (mu*v_in**2.+2.*E_hf)/(mu*v_in**2.)
    elif interaction=='excitation':
        if mu*v_in**2.<=2.*E_hf:
            res = 0.
        else:
            res = (mu*v_in**2.-2.*E_hf)/(mu*v_in**2.)
    return res

def cross_section(v_in,mX,mphi,interaction): # takes v_in in units km/s
    if interaction=='elastic':
        dE = 0.
    elif interaction=='deexcitation':
        dE = E_hf
    elif interaction=='excitation':
        dE = -E_hf
    return form_factor(v_in,mX,interaction)*(mX**2.+mphi**2.)**2./((dE+mX*v_in/c)**2.+mphi**2.)**2.

def integration_over_MB(Ttilde,mX,mphi,interaction): # this is \int v*f(v)*sigma(v)*dv, returns cm/s (with cm^2 from sigma_0)
    v_min = 1e-10*E_hf/mX*c # this is in units km/s
    f = lambda v: 4.*pi*(mX/(2.*pi*k_B*Ttilde*c**2.))**(3./2.)*v**2.*math.exp(-mX*v**2./(2.*k_B*Ttilde*c**2.))
    jacobian = lambda logv: math.exp(logv)
    func = lambda logv: jacobian(logv)*math.exp(logv)*f(math.exp(logv))*cross_section(math.exp(logv),mX,mphi,interaction)
    if interaction=='excitation':
        v_min = max(v_min,math.sqrt(E_hf*(mX+H_mass))/mX*c)
    if v_min>1e10*math.sqrt(k_B*Ttilde/mX*c**2.):
        res = 0.
    else:
        res = 1e5*quad(func,math.log(v_min),math.log( 1e10*math.sqrt(k_B*Ttilde/mX*c**2.) ))[0]
    #  factor 1e5 is conversion from km to cm
    return res

def integration_over_MB_times_Ek(Ttilde,mX,mphi,interaction): # this is \int v*f(v)*sigma(v)*dv*v^2/c^2*mX, returns GeV*cm/s (with cm^2 from sigma_0)
    v_min = 1e-10*E_hf/mX*c # this is in units km/s
    f = lambda v: 4.*pi*(mX/(2.*pi*k_B*Ttilde*c**2.))**(3./2.)*v**2.*math.exp(-mX*v**2./(2.*k_B*Ttilde*c**2.))
    jacobian = lambda logv: math.exp(logv)
    if interaction=='excitation':
        v_min = max(v_min,math.sqrt(E_hf*(mX+H_mass))/(mX*H_mass)*c)
        func = lambda logv: jacobian(logv)*math.exp(logv)*f(math.exp(logv))*cross_section(math.exp(logv),mX,mphi,interaction)*math.exp(logv)**2.*mX/c**2.
    elif interaction=='deexcitation':
        func = lambda logv: jacobian(logv)*math.exp(logv)*f(math.exp(logv))*cross_section(math.exp(logv),mX,mphi,interaction)*(math.exp(logv)**2.*mX/c**2.+E_hf)
    if v_min>1e10*math.sqrt(k_B*Ttilde/mX*c**2.):
        res = 0.
    else:
        res = 1e5*quad(func,math.log(v_min),math.log( 1e10*math.sqrt(k_B*Ttilde/mX*c**2.) ))[0]
    return res