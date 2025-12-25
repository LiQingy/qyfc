'''generate halo mass function
'''
import hmf
import numpy as np
import matplotlib.pyplot as plt

hmfdata = hmf.hmf.MassFunction(z = 0.9, Mmax = 15.5, sigma_8 = 0.811, hmf_model='SMT',transfer_model = 'CAMB', cosmo_params = {'H0':67.4, 'Om0':0.315})
plt.plot(np.log10(hmfdata.m), hmfdata.dndlog10m, '-', c = 'r', lw = 2, label = 'SMT: z = 0.29')
