import numpy as np
from pyscf import gto, scf, mcscf
import json, time, os

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
mol.charge  = -2
mol.spin    = 10
mol.basis   = 'def2-SVP'
mol.verbose = 3
mol.build()

print(f"Atoms:        {mol.natm}")
print(f"Electrons:    {mol.nelectron}")
print(f"Basis funcs:  {mol.nao}")

# Step 1: High-spin UHF
print("\n--- Step 1: High-spin UHF (S=5) ---")
t0 = time.time()
mf = scf.UHF(mol)
mf.max_cycle = 500
mf.conv_tol  = 1e-9
mf.verbose   = 3
mf.kernel()
t1 = time.time()
print(f"Converged: {mf.converged}  Energy: {mf.e_tot:.6f}  Time: {t1-t0:.1f}s")

# Step 2: CASCI on 18-orbital window
print("\n--- Step 2: CASCI(18) ---")
e_mean    = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
occ_total = mf.mo_occ[0] + mf.mo_occ[1]
occ_idx   = np.where(occ_total > 0.5)[0]
virt_idx  = np.where(occ_total <= 0.5)[0]
window18  = sorted(list(occ_idx[-9:]) + list(virt_idx[:9]))

n_act_e = 18
for n in [18,16,20,14]:
    if (mol.nelectron - n) >= 0 and (mol.nelectron - n) % 2 == 0:
        n_act_e = n
        break

print(f"Active electrons: {n_act_e}")
mo_avg = (mf.mo_coeff[0] + mf.mo_coeff[1]) / 2
mc = mcscf.CASCI(mf, 18, n_act_e)
mc.verbose = 3
mo = mc.sort_mo(window18, mo_coeff=mo_avg, base=0)
mc.kernel(mo)
t2 = time.time()

casdm1    = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
no_cas, _ = np.linalg.eigh(casdm1)
no_cas    = np.sort(no_cas)[::-1]
eps       = 1e-12
n_clip    = np.clip(no_cas/2, eps, 1-eps)
s_i       = -(n_clip*np.log(n_clip) + (1-n_clip)*np.log(1-n_clip))
frac      = int(np.sum((no_cas > 0.02) & (no_cas < 1.98)))

print(f"CASCI converged: {mc.converged}")
print(f"CASCI energy:    {mc.e_tot:.6f}")
print(f"NOONs: {np.round(no_cas,4)}")
print(f"s_i:   {np.round(s_i,4)}")
print(f"Estimated n_active: {frac}")
print(f"CASCI time: {t2-t1:.1f}s")
print(f"Total time: {t2-t0:.1f}s")

os.makedirs(os.path.expanduser('~/activeml/results/fe2s2_test'), exist_ok=True)
json.dump({
    'converged_uhf' : bool(mf.converged),
    'converged_casci': bool(mc.converged),
    'n_electrons'   : mol.nelectron,
    'n_basis'       : mol.nao,
    'no_occ'        : no_cas.tolist(),
    's_i'           : s_i.tolist(),
    'n_active_est'  : frac,
    'time_uhf'      : round(t1-t0,1),
    'time_casci'    : round(t2-t1,1),
}, open(os.path.expanduser('~/activeml/results/fe2s2_test/Fe2S2_test.json'),'w'), indent=2)
print("Saved.")
