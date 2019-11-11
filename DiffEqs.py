import math
import numpy as np
import matplotlib.pyplot as plt
pi=math.pi

c = 2.998e5
# Boltzmann constant in GeV/K
k_B = 8.61733e-14
T_hf = 0.068 # temperature of hyperfine transition
E_hf = 5.87433e-15 # in GeV
A10 = 2.85e-15
H_mass = 0.9388 # in GeV

from CrossSection import integration_over_MB,integration_over_MB_times_Ek

# scattering rate for hydrogen-hydrogen (see Eq. 9 in the paper)
def kHH10(z): # result is in units cm^3/s
    return 3.1e-11*T_gas(z)**0.357*np.exp(-32./T_gas(z))

# scattering rate for hydrogen with free electron (see Eq. 9 in the paper)
def keH10(z): # result is in units cm^3/s
    return np.exp(   -9.607+0.5*np.log(T_gas(z))*np.exp(-np.log(T_gas(z))**4.5/1800.)   )

# CMB temperature as function of redshift
def T_CMB(z):
    return 2.726*(1.+z)

# hydrogen gas temperature as function of redshift, assuming adiabatic cooling
def T_gas(z):
    return (1.+z)**2./201.**2.*T_CMB(200.)*(z<200.)+T_CMB(z)*(z>=200.)+(z<20.)*(np.exp(20.-z)-1.)/100.

# effective temperature of dark matter-hydrogen interactions, see section 2.3 in the paper
def T_tilde(z,mX):
    return T_gas(z)/H_mass/(1./mX+1./H_mass)

# the hydrogen number denstiy
def hydrogen_number_density(z): # result is in units cm^-3
    return 1.877e-7*(1.+z)**3.

# the dark matter number denstiy
def DM_number_density(z,mX): # mX in GeV please; result is in units cm^-3
    # 0.2589*rho_c
    return 1.252e-6/mX*(1.+z)**3.

# fraction of free electrons
def e_fraction(z): # units K; redshift
    T_ionize = 1.579e5 # ionization energy in Kelvin
    n = hydrogen_number_density(z)/0.75*(0.75+0.125)
    return np.sqrt(  1./n*(1.79987e10*T_gas(z))**1.5*np.exp(-T_ionize/T_gas(z))  )

# the spin temperature
def T_spin(z,mX,mphi,sigma_0,DMfraction=1.,Ttilde=None):
    if Ttilde==None:
        Ttilde = T_tilde(z,mX)
    p10_collision = (kHH10(z)+e_fraction(z)*keH10(z))*hydrogen_number_density(z)
    p10_alpha = A10/T_hf*(z<17.)*(np.exp(17.-z)-1.)**3./200.
    p10_DM = DMfraction*DM_number_density(z,mX)*sigma_0*integration_over_MB(Ttilde,mX,mphi,'deexcitation')
    p01_DM = DMfraction*DM_number_density(z,mX)*sigma_0*integration_over_MB(Ttilde,mX,mphi,'excitation')
    nominator = T_CMB(z)/T_hf*A10 + np.exp(-T_hf/T_gas(z))*(p10_collision+p10_alpha) + p01_DM
    denominator = A10*(1.+T_CMB(z)/T_hf) + (p10_collision+p10_alpha) + p10_DM
    rhs = nominator/denominator
    return -T_hf/np.log(rhs)

# the heating power of dark matter from interactions with hydrogen
def dark_matter_heating_per_second(z,mX,mphi,g_N,g_x,Ttilde=None):# returns mean heating of dark matter particle in units GeV/s
    if Ttilde==None:
        Ttilde = T_tilde(z,mX)
    sigma_0 = g_N**2.*g_x**2.*mX**2./(4.*pi*6.5e25*(mX**2.+mphi**2.)**2.)
    energy_by_excitations_per_second = sigma_0*hydrogen_number_density(z)*integration_over_MB_times_Ek(Ttilde,mX,mphi,'excitation')
    energy_by_deexcitations_per_second = sigma_0*hydrogen_number_density(z)*integration_over_MB_times_Ek(Ttilde,mX,mphi,'deexcitation')
    return energy_by_excitations_per_second+energy_by_deexcitations_per_second

# returns the coupling constant that produces a specific spin temperature at some redshift, for specific model parameters
def g_N_limit(z,T_spin,mX,mphi,DMfraction=1.,Ttilde=None):
    if Ttilde==None:
        Ttilde = T_tilde(z,mX)
    p10_collision = (kHH10(z)+e_fraction(z)*keH10(z))*hydrogen_number_density(z)
    p10_alpha = 0.
    p10_DM_over_sigma_0 = DMfraction*DM_number_density(z,mX)*integration_over_MB(Ttilde,mX,mphi,'deexcitation')
    p01_DM_over_sigma_0 = DMfraction*DM_number_density(z,mX)*integration_over_MB(Ttilde,mX,mphi,'excitation')
    a_term = T_CMB(z)/T_hf*A10 + np.exp(-T_hf/T_gas(z))*(p10_collision+p10_alpha)
    b_term = A10*(1.+T_CMB(z)/T_hf) + (p10_collision+p10_alpha)
    c_term = p01_DM_over_sigma_0
    d_term = p10_DM_over_sigma_0
    y_term = np.exp(-T_hf/T_spin)
    sigma_0 = (a_term-b_term*y_term)/(-c_term+d_term*y_term)
    if DMfraction<0.3:
        g_x_squared = 1.
    else:
        # limit from Bullet Cluster
        g_x_squared = math.sqrt(1.8e-4*mX**3.)
    g_N_squared = sigma_0*4.*pi*6.5e25*(mX**2.+mphi**2.)**2./(mX**2.*g_x_squared)
    return math.sqrt(g_N_squared)

# returns the coupling constant that produces a specific spin temperature at redshift z=17, also including heating of the different gases
def g_N_limit_iteration_z17(T_spin,mX,mphi,DMfraction=1.,Tx_init_at_z0=0.):
    Ttilde17 = T_tilde(17.,mX)
    for iteration in range(20):
        g_N = g_N_limit(17.,T_spin,mX,mphi,DMfraction=DMfraction,Ttilde=Ttilde17)
        Tx_at_z0 = Tx_init_at_z0
        Ttilde = T_tilde(100.,mX)
        for i_z in range(100,16,-1):
            z = float(i_z)
            energy_added_per_z = dark_matter_heating_per_second(z,mX,mphi,g_N,1.,Ttilde=Ttilde)/(math.sqrt(.3)*2e-18*(1.+z)**(5./2.))
            Tx_at_z0 += 2./3.*energy_added_per_z/k_B/(1.+z)**2.
            Tx = Tx_at_z0*(1.+z)**2.
            Ttilde = (T_gas(z)/H_mass+Tx/mX)/(1./H_mass+1./mX)
        Ttilde17 = Ttilde
    g_N = g_N_limit(17.,T_spin,mX,mphi,DMfraction=DMfraction,Ttilde=Ttilde17)
    z_vec = []
    Ttilde_vec = []
    Tx_vec = []
    Tx_at_z0 = Tx_at_z0 = Tx_init_at_z0
    Ttilde = T_tilde(1000.,mX)
    for i_z in range(1000,9,-1):
        z = float(i_z)
        energy_added_per_z = dark_matter_heating_per_second(z,mX,mphi,g_N,1.,Ttilde=Ttilde)/(math.sqrt(.3)*2e-18*(1.+z)**(5./2.))
        Tx_at_z0 += 2./3.*energy_added_per_z/k_B/(1.+z)**2.
        Tx = Tx_at_z0*(1.+z)**2.
        Ttilde = (T_gas(z)/H_mass+Tx/mX)/(1./H_mass+1./mX)
        z_vec.append(z)
        Ttilde_vec.append(Ttilde)
        Tx_vec.append(Tx)
    return g_N,z_vec,Ttilde_vec,Tx_vec
