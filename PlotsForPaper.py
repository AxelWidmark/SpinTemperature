import math
import numpy as np
import matplotlib.pyplot as plt
pi=math.pi
from scipy.interpolate import interp1d


from CrossSection import integration_over_MB
from DiffEqs import T_tilde

# specify font for plots
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams["mathtext.fontset"] = "dejavusans"
rcParams['font.sans-serif'] = "dejavusans"



# makes Fig. 5 in the paper
def plot_gNcontour():
    from DiffEqs import T_spin
    f = plt.figure(figsize=(6.,4.5))
    from DiffEqs import g_N_limit,g_N_limit_iteration_z17,dark_matter_heating_per_second,H_mass,T_gas,T_tilde,k_B,E_hf,T_hf
    mXvec = np.logspace(-6.,-2.,51)
    mphivec = np.logspace(-12.,-8.,51)
    DMfraction = 0.1
    xv, yv = np.meshgrid(mXvec, mphivec)
    g_N_matrix = np.zeros((len(mXvec),len(mphivec)))
    Ttilde_matrix = np.zeros((len(mXvec),len(mphivec)))
    rel_Ttilde_matrix = np.zeros((len(mXvec),len(mphivec)))
    dTs_dz_matrix = np.zeros((len(mXvec),len(mphivec)))
    for i in range(len(mXvec)):
        for j in range(len(mphivec)):
            print(i,j)
            mX = mXvec[i]
            mphi = mphivec[j]
            g_N,z_vec,Ttilde_vec,Tx_vec = g_N_limit_iteration_z17(3.,mX,mphi,DMfraction=DMfraction)
            Ttilde17 = np.interp(17.,z_vec[::-1],Ttilde_vec[::-1])
            print(Ttilde)
            g_N_matrix[i][j] = g_N
            Ttilde17_matrix[i][j] = Ttilde17
            #rel_Ttilde_matrix[i][j]  = Ttilde/T_tilde(17.,mX)
            #rel_Ttilde_matrix[i][j]  = Ttilde/T_hf
            sigma_0 = g_N**2.*1.**2.*mX**2./(4.*pi*6.5e25*(mX**2.+mphi**2.)**2.)
    np.savez('./matrices_50by50',g_N_matrix=g_N_matrix,Ttilde17_matrix=Ttilde17_matrix)
    print(g_N_limit(17.,3.,1e-3,1e-9,DMfraction=DMfraction))
    plt.contourf(np.log10(mphivec)+9.,np.log10(mXvec)+3.,np.log10(g_N_matrix),levels=range(-17,-8,1))#,cmap=plt.cm.tab10)
    plt.xticks(range(-3,2),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
    plt.yticks(range(-3,2),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
    clb = plt.colorbar()
    clb.set_label('log$_{10}(g_N)$')
    plt.xlabel('$m_V$ (eV)')
    plt.ylabel('$m_\chi$ (MeV)')
    plt.plot([],[],'k-',label=r'$\epsilon(\tilde{T})$')
    plt.contourf(np.log10(mphivec)+9.,np.log10(mXvec)+3.,[[np.log10(mX/mphi) for mphi in mphivec] for mX in mXvec],levels=[8.,np.inf],colors=['w'])#,cmap=plt.cm.tab10)
    plt.tick_params(axis='y', which='both')
    plt.tight_layout()
    plt.savefig('./Figures/gNcontour.pdf')
    plt.show()

    
  
# makes Figures 2 and 3 in the paper
def plot_formfactor_and_cross():
    from CrossSection import form_factor
    from CrossSection import cross_section
    E_hf = 5.87433e-15
    mX=1e-3
    f = plt.figure(figsize=(5,3.5))
    v_vec = np.logspace(-1.,1.,1000)
    plt.loglog(v_vec,[form_factor(v,mX,'deexcitation') for v in v_vec],'k--',linewidth=2,label='Deexcitation')
    plt.loglog(v_vec,[form_factor(v,mX,'elastic') for v in v_vec],'k',linewidth=2,label='Elastic')
    v_vec = np.logspace(max(-1.,np.log10(math.sqrt(E_hf))),1.,1000)
    plt.loglog(v_vec,[form_factor(v,mX,'excitation') for v in v_vec],'k:',linewidth=2,label='Excitation')
    plt.xlim([1e-1,1e1])
    plt.ylim([1e-2,1e2])
    plt.xlabel('Collisional velocity (km/s)')
    plt.ylabel('Form factor')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Figures/form_factor.pdf')
    plt.show()
    f = plt.figure(figsize=(5,3.5))
    v_vec = np.logspace(-3.,2.,1000)
    mphi = 1e-7
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'deexcitation') for v in v_vec],'r--',linewidth=2)
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'elastic') for v in v_vec],'r',linewidth=2)
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'excitation') for v in v_vec],'r:',linewidth=2)
    mphi = 1e-9
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'deexcitation') for v in v_vec],'k--',linewidth=2,label='Deexcitation')
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'elastic') for v in v_vec],'k',linewidth=2,label='Elastic')
    plt.loglog(v_vec,[cross_section(v,mX,mphi,'excitation') for v in v_vec],'k:',linewidth=2,label='Excitation')
    plt.text(4.,1e20,'$m_V = 1$ eV')
    plt.text(2.,1e14,'$m_V = 100$ eV',color='r')
    plt.xlim([1e-3,1e2])
    plt.ylim([1e13,1e31])
    plt.xlabel('Collisional velocity (km/s)')
    plt.ylabel('Cross section enhancement')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Figures/cross_section.pdf')
    plt.show()

    
    
# makes Fig. 1 in the paper
def plot_Tspin():
    mX = 1e-3
    mphi = 1e-10
    from DiffEqs import T_CMB,T_gas,T_spin
    f = plt.figure(figsize=(6,5))
    zvec = np.logspace(1.,3.,1001.)
    plt.loglog(zvec,T_CMB(zvec),'b',label='$T_\gamma$')
    plt.loglog(zvec,[T_gas(z) for z in zvec],'r',label='$T_K$')
    plt.loglog(zvec,[T_spin(z,mX,mphi,0.) for z in zvec],'k--')
    plt.plot([],[],'k--',label='$T_s$')
    plt.xlim([1e1,1e3])
    plt.ylim([1e0,5e3])
    plt.xlabel('$z$')
    plt.ylabel('Temperature (K)')
    plt.legend(loc=2)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.savefig('./Figures/Tspin.pdf')
    plt.show()

    
    
    
# makes Fig. 4 in the paper
def plot_Tspin2():
    from DiffEqs import T_CMB,T_gas,T_spin,g_N_limit,H_mass,g_N_limit_iteration_z17
    DMfraction = 0.1
    f = plt.figure(figsize=(6,5))
    plt.loglog([],[],'b',label=r'$T_\gamma$')
    plt.loglog([],[],'r',label=r'$T_K$')
    zvec = np.logspace(1.,3.,1001.)
    mX = 1e-3
    mphi = 1e-9
    g_N = g_N_limit(17.,3.,mX,mphi,DMfraction=DMfraction)
    print(g_N,'\n')
    sigma0 = g_N**2.*1.**2.*mX**2./(4.*pi*6.5e25*(mX**2.+mphi**2.)**2.)
    plt.loglog(zvec,[T_spin(z,mX,mphi,sigma0,DMfraction=DMfraction) for z in zvec],color='k',linestyle='-',label=r'$T_s$, $\chi$ cold, no heating')
    g_N,z_vec,Ttilde_vec,Tx_vec = g_N_limit_iteration_z17(3.,mX,mphi,DMfraction=DMfraction,Tx_init_at_z0=0.)
    print(Tx_vec[-1]/(1+z_vec[-1])**2.)
    print(g_N,'\n')
    def T_tilde(z):
        return np.interp(z,z_vec[::-1],Ttilde_vec[::-1])
    sigma0 = g_N**2.*1.**2.*mX**2./(4.*pi*6.5e25*(mX**2.+mphi**2.)**2.)
    plt.loglog(zvec,[T_spin(z,mX,mphi,sigma0,DMfraction=DMfraction,Ttilde=T_tilde(z)) for z in zvec],color='k',linestyle='--',label=r'$T_s$, $\chi$ cold initially')
    g_N,z_vec,Ttilde_vec,Tx_vec = g_N_limit_iteration_z17(3.,mX,mphi,DMfraction=DMfraction,Tx_init_at_z0=50.*Tx_vec[-1]/(1+z_vec[-1])**2.)
    print(Tx_vec[-1]/(1+z_vec[-1])**2.)
    print(g_N,'\n')
    def T_tilde(z):
        return np.interp(z,z_vec[::-1],Ttilde_vec[::-1])
    sigma0 = g_N**2.*1.**2.*mX**2./(4.*pi*6.5e25*(mX**2.+mphi**2.)**2.)
    plt.loglog(zvec,[T_spin(z,mX,mphi,sigma0,DMfraction=DMfraction,Ttilde=T_tilde(z)) for z in zvec],color='k',linestyle=':',label=r'$T_s$, $\chi$ warm initially')
    plt.loglog(zvec,T_CMB(zvec),'b')
    plt.loglog(zvec,[T_gas(z) for z in zvec],'r')
    plt.xlim([1e1,1e3])
    plt.ylim([1e0,5e3])
    plt.xlabel('$z$')
    plt.ylabel('Temperature (K)')
    plt.legend(loc=2)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.savefig('./Figures/Tspin2.pdf')
    plt.show()
