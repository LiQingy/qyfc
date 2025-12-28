import numpy as np
from scipy.interpolate import interp1d
import scipy.linalg as la
import emcee
import corner
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from mcfit import P2xi

import os
import sys
sys.path.append("/home/qyli/work2025/densityfield/project2/xi")
import stamcmc
importlib.reload(stamcmc)

import h5py


def log_prior(theta):
    B, s0, gamma, N, sigma, sm = theta
    if not (-1 < B < 1): 
        return -np.inf
    if not (2 < s0 < 20):
        return -np.inf
    if not (0 < gamma < 10):
        return -np.inf
    if not (0.0001 < N < 0.8):
        return -np.inf
    if not (3 < sigma < 16):
        return -np.inf
    if not (90 < sm < 118):
        return -np.inf
    return 0.0

def log_likelihood(theta, s, xi_obs, Cinv):
    B, s0, gamma, N, sigma, sm = theta
    model = (B + (s/s0)**(-gamma) + N/(2*np.pi*sigma**2)**0.5 * np.exp(-(s-sm)**2/2/sigma**2))*s**2
    diff = xi_obs - model
    chi2 = diff.dot(Cinv).dot(diff)
    return -0.5 * chi2

def log_posterior(theta, s, xi_obs, Cinv):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, s, xi_obs, Cinv)
    return lp+ll


def run_mcmc(s, xi_obs, Cinv, bounds,
             nwalkers=2000, nsteps=5000, burnin=1000, threads=4):
    
    p0 = np.zeros((nwalkers, len(bounds)))
    for i in range(len(bounds)):
        lower, upper = bounds[i]
        p0[:, i] = np.random.uniform(lower, upper, nwalkers)

    ndim = p0.shape[1]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=(s, xi_obs, Cinv),
                                    threads=threads)

    print("Start MCMC: nwalkers=", nwalkers, " nsteps=", nsteps)
    sampler.run_mcmc(p0, nsteps, progress=True)
    flat_samples = sampler.get_chain(discard=burnin, thin = 15, flat=True)
    
    log_prob = sampler.get_log_prob(discard=burnin, thin = 15, flat=True)  # shape (Nsamples,)
    imax = np.argmax(log_prob)
    best_theta = flat_samples[imax]
    print("chi2 for best parameter", log_prob[imax]/(-0.5))
    
    return flat_samples, best_theta


if __name__ == "__main__":

    filepath = '/home/qyli/work2025/densityfield/project2/xi/data/'

    dnames = ['9tianM_SpecNGC']
    for dname in dnames:
        fnames = np.array([
        '%s/stat_%s_mlim12_Ngrid512_Ms14.75_z0.0-0.4.hdf5'%(dname, dname),
        # '%s/stat_%s_mlim12_Ngrid350_sigma1.0_z0.4-0.6_CIC_Nmock50_linearbin.hdf5'%(dname, dname),
        # '%s/stat_%s_mlim12.5_Ngrid512_sigma1.0_z0.6-0.8_CIC_Nmock50_linearbin.hdf5'%(dname, dname),
        # '%s/stat_%s_mlim13_Ngrid600_sigma1.0_z0.8-1.0_CIC_Nmock50_linearbin.hdf5'%(dname, dname)
        ])
        zlabs = ['z0.0-0.4']
        for fname,zlab in zip(fnames, zlabs):
            dh5 = h5py.File(filepath+fname, 'r')
            rx = dh5['r_bin_corr'][:]
            xi_corr = dh5['xi_bin_corr'][:]
            jk_xi = dh5['jackknife_xi'][:]

            idx_ = (rx > 40) & (rx < 135)
            s_fit = rx[idx_]  
            xi_fit = xi_corr[idx_]*s_fit**2
            jk_fit = jk_xi[:,idx_]*s_fit**2

            cov, Cinv, xi_mean = stamcmc.compute_covariance_and_inverse(jk_fit, Hartlap = True, jackknife = True)

            n_cpu = os.cpu_count()
            bounds = [[-0.001, -0.00001], [2, 5], [2, 5], [0.0002, 0.06], [4, 12], [95, 110]]
            flat_samples, best_theta = run_mcmc(s_fit, xi_fit, Cinv, bounds,
                    nwalkers=64, nsteps=10000, burnin=1000, threads=n_cpu)

            np.save("../xi/data/%s/mcmc_%s_%s.npy"%(dname, dname, zlab), flat_samples)

            theta = np.median(flat_samples, axis=0)
            print("parameters under minimal chi2", best_theta)
            print("parameters under median value", theta)

            chi0 = log_likelihood(best_theta, s_fit, xi_fit, Cinv)
            print("chi2 corresponding to the best parameters is", chi0/-0.5)
            chi0 = log_likelihood(theta, s_fit, xi_fit, Cinv)
            print("chi2 corresponding to the median parameters is", chi0/-0.5)
