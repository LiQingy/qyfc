import numpy as np 
import h5py                                                       

def cal_BCGinfo(dgroup, d2, d1, path_output_file):
    '''
    calculate BCG information
    
    Note:
    =========
    save _BCGinfo file:
    1. group seqid
    2. brightest galaxy seqid
    3. massive galaxy seqid
    '''
    Ngroup = dgroup.shape[0]
    f_BCGinfo = np.zeros((Ngroup,3), dtype = np.int64)
    f_BCGinfo[:,0] = np.arange(1,Ngroup+1,1)
    igal0 = 0
    igal1 = 0
    for i in range(Ngroup):
        groupid = i+1
        igal1 = np.int64(igal0+dgroup[i,1])
        galid = np.int64(d2[igal0:igal1,2])
        idmax_lum = np.where((d1[galid-1,2] == 1))[0]
        idmax_mass = np.where((d1[galid-1,3] == 1))[0]

        f_BCGinfo[i,1] = galid[idmax_lum]
        f_BCGinfo[i,2] = galid[idmax_mass]

        igal0 = igal1

    np.savetxt(path_output_file, f_BCGinfo, fmt = '%i')

if __name__ == '__main__':
    dgroup = np.loadtxt('./outputnew/tng_group')
    d2 = np.loadtxt('./outputnew/itng_2')
    d1 = np.loadtxt('./outputnew/itng_1')
    path_output_file = './outputnew/tng_BCGinfo'

    cal_BCGinfo(dgroup, d2, d1, path_output_file)