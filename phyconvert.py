from astropy.cosmology import Planck18 as cosmo
import numpy as np

def r200c_from_mass_z(M200c, z):
    """
    General R200c computation

    M200c : Msun
    return : Mpc
    """

    rho_c = cosmo.critical_density(z).to('Msun/Mpc^3').value

    R200c = (3 * M200c / (4 * np.pi * 200 * rho_c))**(1/3)

    return R200c
