import numpy as np 
import healpy as hp
from pandas import value_counts
import healpy as hp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np 


def generate_1d_dist(data, require_num, hlim, nbin = 100):
    '''generate points following a 1d distribution
    '''
    counts, bin_edges = np.histogram(data, bins=nbin, range = (hlim[0], hlim[1]), density=True)

    cdf = np.cumsum(counts) * np.diff(bin_edges)[0] #cumulative possibility
    cdf = cdf / cdf[-1]  

    random_uniform = np.random.rand(require_num)  
    random_values = np.interp(random_uniform, cdf, bin_edges[1:]) 
    return random_values


def find_index(b, a):
	'''find the b in a, return index of a
	'''
	dict_a = {v: i for i, v in enumerate(a)}
	indx_a = np.int64([dict_a[v] for v in b])
	return indx_a


def get_d(z, dist_type = 'luminosity'):
    '''get comoving distance based on cosmo
    '''
    if dist_type == 'luminosity':
        return cosmo.luminosity_distance(z).value #Mpc
    elif dist_type == 'comoving':
        return cosmo.comoving_distance(z).value #Mpc

def get_zd_interpl(dist, dist_type = 'luminosity'):
    from scipy.interpolate import interp1d
    zlin = np.linspace(1e-6, 6, 6000)
    distlin = get_d(zlin, dist_type)
    f = interp1d(distlin, zlin, kind='quadratic')
    return f(dist)


def save_to_hdf5(filename, dataset_name, data):
    """save data to HDF5"""
    with h5py.File(filename, "w") as f:
        grp = f.create_group(dataset_name)
        for name in data.dtype.names:
            grp.create_dataset(name, data=data[name], compression="gzip")

def interpolate_func()
    from scipy.interpolate import interp1d
    interp = interp1d(z_cos.value, z_vals, kind='linear', bounds_error=False, fill_value=np.nan)
    redshift = interp(r_comoving)


def cal_galarea(ra,dec):
    '''
    calculate the area of source coverage on all sky

    return: source area [deg^2]
    '''

    nside = 512
    totalpixel = hp.nside2npix(nside)

    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra))
    pxlc = pd.value_counts(pxl)

    gal_area = pxlc.shape[0] / totalpixel * (4*180**2/np.pi)
    print("the cover of galaxy region is %s dge2" %gal_area)
    
    return gal_area


def cal_sky_coverage( ra, dec, nside=512, autoplot=False ):
    '''
    calculate the sky coverage

    Parameter:
    ----------
    ra,
    dec,
    nside: resolution of maps, Defalut: 512
    autoplot: plot sky coverage maps, Defalut: False

    return:
    -------
    gal_area: sky coverage [deg^2]

    '''

    print('===> size of pixel in unit of arcmin = ',np.sqrt( 41253.*60.*60./hp.nside2npix(nside) ))

    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl) #count the amount of same healpix pixel numbers 

    if autoplot: # plot the galaxy number over density per pixel
        colorpix = np.zeros(hp.nside2npix(nside))
        colorpix[pxlc.index] = pxlc.values
        plt.figure()
        hp.mollview(colorpix, title='Galaxy Number Counts',rot=(270.,0.,0.),xsize =2000)
        hp.graticule(dpar=30)

    totalpixel = hp.nside2npix(nside)
    gal_area = pxlc.shape[0] / totalpixel * (4*180**2/np.pi)
    print("the cover of galaxy region is %s dge2" %gal_area)

    return gal_area






def position_xyz(cosmo,ra,dec,z):
    '''
    Convert ra,dez,z in coordinate into x,y,z
    '''
    import math
    degs = math.pi / 180
    dist = (cosmo.comoving_distance(z)).value * 1000 #kpc
    # dist = comvdis(z) * 1000 #kpc
    x1 = ra * degs 
    x2 = dec * degs
    unit = 1
    rx = (dist * math.cos(x2) * math.cos(x1) * unit)
    ry = (dist * math.cos(x2) * math.sin(x1) * unit)
    rz = (dist * math.sin(x2) * unit)
    return rx,ry,rz #unit: kpc

def cal_2Ddis_pLoS(poscen,posgal):
    '''
    calculate penticular distance along light of sight at projected plane
    the origin of coordinate is at the Earth
    '''
    import math
    rgal = math.sqrt(posgal[0]**2 + posgal[1]**2 + posgal[2]**2)
    rcen = math.sqrt(poscen[0]**2 + poscen[1]**2 + poscen[2]**2)

    r1 = abs(rgal - rcen)
    r0 = math.sqrt((posgal[0] - poscen[0])**2 + (posgal[1] - poscen[1])**2 + (posgal[2] - poscen[2])**2)
    r0r = math.sqrt(r0**2 - r1**2)

    return r0r #unit: kpc

def cal_rp(ag_redz,c_redz,ag_ra,ag_dec,c_ra,c_dec):
    '''
    perpendicular distance
    '''
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
    
    dr_gal = cosmo.comoving_distance(ag_redz).value * 0.674 #Mpc/h
    dr_clu = cosmo.comoving_distance(c_redz).value * 0.674 #Mpc/h

    cgal = SkyCoord(ra=ag_ra*u.degree, dec=ag_dec*u.degree, distance=dr_gal)
    cclu = SkyCoord(ra=c_ra*u.degree, dec=c_dec*u.degree, distance=dr_clu)
    cgal_x = cgal.cartesian.x
    cgal_y = cgal.cartesian.y
    cgal_z = cgal.cartesian.z

    cclu_x = cclu.cartesian.x
    cclu_y = cclu.cartesian.y
    cclu_z = cclu.cartesian.z

    l = np.array([cgal_x+cclu_x, cgal_y+cclu_y, cgal_z+cclu_z]).T / 2
    s = np.array([cclu_x - cgal_x, cclu_y - cgal_y, cclu_z - cgal_z]).T
    r_pi = np.sum(l*s,axis = 1) / np.sqrt(np.sum(l**2, axis = 1)) 
    r_p = np.sqrt(np.sum(s**2,axis = 1) - r_pi**2)

    # aperature distance
    # cdud = SkyCoord(cra, cdec, unit="deg")
    # cc = SkyCoord(ag_ra, ag_dec, unit="deg")
    # sep = cdud.separation(cc)
    # d_A = cosmo.comoving_distance(credz) * 0.674 #Mpc/h
    # d_r = (sep * d_A).to(u.Mpc, u.dimensionless_angles()).value #to be a comoving distance
    
    return cal_rp


def cal_Vmax(zBGG, AMBGG, zlim, mlim, galarea):
    '''
    Calculate Vmax in a magnitude limited sample
    using the method in Yang et al. 2021
    
    Parameters:
    ============
    zBGG: redshift for BGG
    AMBGG: absolute magnitude for BGG
    zlim: redshift bin
    mlim: apparent magnitude limit
    galarea: sky coverage
    
    Returns:
    =========
    Vmax
 
    '''
    from astropy.cosmology import FlatLambdaCDM
    import numpy as np
    
    cosmo_para = {'H0':67.4, 'Om0':0.315}
    h = cosmo_para['H0'] / 100
    cosmo = FlatLambdaCDM(H0 = cosmo_para['H0'], Om0 = cosmo_para['Om0'])
    Vratio = galarea / ((180/np.pi)**2*4*np.pi)

    dlimlow = cosmo.comoving_distance(zlim[0]).value#Mpc
    Vlowmax = 4/3*np.pi*dlimlow**3 * Vratio
    
    dmax = 10**((mlim - AMBGG - 25) / 5) #maximum distance from limited luminosity
    dmax_como = dmax / (1 + zBGG)

    dlimup = cosmo.comoving_distance(zlim[1]).value
    Vupmax = np.zeros(zBGG.shape[0])
    Vupmax[dmax_como < dlimup] = 4/3*np.pi*(dmax_como[dmax_como < dlimup])**3 * Vratio
    Vupmax[dmax_como >= dlimup] = 4/3*np.pi*(dlimup)**3 * Vratio

    Vmax = Vupmax - Vlowmax #Mpc

    return Vmax





