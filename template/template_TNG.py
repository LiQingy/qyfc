import illustris_python as il
import numpy as np

def tng_result(basePath = '/home/cossim/IllustrisTNG/TNG300-1/output'):
    '''trace the group mass of the most massive subhalos
    '''
    
    groupprop = il.groupcat.loadHalos(basePath, 99, fields=['GroupMass','GroupFirstSub' ])
    grpmass_tng300 = np.log10(groupprop['GroupMass'][:] / 0.6774 * 1e10) #log Msun
    idxsel = (grpmass_tng300 > 14) & (grpmass_tng300 < 14)
    idsseq = np.arange(len(grpmass_tng300))[idxsel]

    GroupFirstSub = il.groupcat.loadHalos(basePath, 99, fields=['GroupFirstSub'])
    
    #save redshifts
    print("save redshifts......")
    basePath = '/home/cossim/IllustrisTNG/TNG300-1/output'
    redz = np.zeros(99)
    for ii in range(99, 0, -1):
        Header = il.groupcat.loadHeader(basePath, ii)
        redz[99-ii] = Header['Redshift']

    #trace subhalos
    print("trace subhalo merger tree......")
    group_evo_nrs = np.zeros((len(idsseq), 99))
    group_evo_redz = np.zeros((len(idsseq), 99))
    fields = ['SubhaloMass','SubfindID','SnapNum', 'SubhaloGrNr']
    for ix,ii in enumerate(idsseq):
        tree = il.sublink.loadTree(basePath, 99, GroupFirstSub[ii], fields=fields, onlyMPB=True)
        nsnap = len(tree['SnapNum'][:])
        group_evo_redz[ix, :nsnap] = redz[99 - tree['SnapNum'][:]]
        group_evo_nrs[ix, :nsnap] = tree['SubhaloGrNr'][:]

    #select the main halo that subhalo belongs
    print("search group information......")
    group_evo_mass = np.zeros((len(idsseq), 99))
    for ii in range(99,20,-1):
        groupprop = il.groupcat.loadHalos(basePath, ii, fields=['GroupMass','GroupFirstSub'])
        grp_ids = np.int64(group_evo_nrs[:,99-ii])
        group_evo_mass[:,99-ii] = groupprop['GroupMass'][grp_ids]
        print(ii)
    return group_evo_nrs, group_evo_redz, group_evo_mass

if __name__ == '__main__':
    group_evo_nrs, group_evo_redz, group_evo_mass = tng_result(basePath = '/home/cossim/IllustrisTNG/TNG300-1/output')
