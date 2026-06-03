#!/bin/bash
#SBATCH --job-name=fe2s2_casscf
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_casscf_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_casscf_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Started: $(date)"
python3 -c "
import numpy as np, time
from pyscf import gto, scf, mcscf

mol = gto.Mole()
mol.atom = '''
Fe  0.000   1.345   0.000
Fe  0.000  -1.345   0.000
S   1.600   0.000   0.000
S  -1.600   0.000   0.000
S   0.000   2.800   1.500
H   0.000   3.900   1.500
S   0.000   2.800  -1.500
H   0.000   3.900  -1.500
S   0.000  -2.800   1.500
H   0.000  -3.900   1.500
S   0.000  -2.800  -1.500
H   0.000  -3.900  -1.500
'''
mol.charge=-2; mol.spin=0
mol.basis='def2-SVP'; mol.verbose=0
mol.build()

mf = scf.UHF(mol)
mf.max_cycle=500; mf.kernel()
print(f'UHF: E={mf.e_tot:.4f} S2={mf.spin_square()[0]:.3f}')

e_mean = (mf.mo_energy[0]+mf.mo_energy[1])/2
occ_total = mf.mo_occ[0]+mf.mo_occ[1]
occ_idx  = np.where(occ_total>0.5)[0]
virt_idx = np.where(occ_total<=0.5)[0]
window14 = sorted(list(occ_idx[-7:])+list(virt_idx[:7]))

t1 = time.time()
mc = mcscf.CASSCF(mf, 14, 14)
mc.verbose = 3
mc.max_memory = 12000
mo_avg = (mf.mo_coeff[0]+mf.mo_coeff[1])/2
mo = mc.sort_mo(window14, mo_coeff=mo_avg, base=0)
mc.kernel(mo)
t2 = time.time()

casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
no_cas,_ = np.linalg.eigh(casdm1)
no_cas = np.sort(no_cas)[::-1]
eps = 1e-12
n_clip = np.clip(no_cas/2, eps, 1-eps)
s_i = -(n_clip*np.log(n_clip)+(1-n_clip)*np.log(1-n_clip))
frac = int(np.sum((no_cas>0.02)&(no_cas<1.98)))

print(f'CASSCF(14,14): converged={mc.converged} time={t2-t1:.1f}s')
print(f'NOONs: {np.round(no_cas,4)}')
print(f's_i:   {np.round(s_i,4)}')
print(f'n_active: {frac}')
"
echo "Finished: $(date)"
