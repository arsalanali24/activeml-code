"""
Fe2S2 broken-symmetry test
Antiferromagnetic state: Fe1=spin up, Fe2=spin down
"""
import numpy as np
from pyscf import gto, scf, mcscf
import json, time, os

# ── MOLECULE ──────────────────────────────────────────────────
mol_hs = gto.Mole()
mol_hs.atom = '''
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
mol_hs.charge = -2
mol_hs.spin   = 10
mol_hs.basis  = 'def2-SVP'
mol_hs.verbose = 0
mol_hs.build()

# ── STEP 1: HIGH-SPIN UHF (starting point) ────────────────────
print("Step 1: High-spin UHF for initial orbitals...")
t0  = time.time()
mf_hs = scf.UHF(mol_hs)
mf_hs.max_cycle = 500
mf_hs.conv_tol  = 1e-9
mf_hs.verbose   = 0
mf_hs.kernel()
print(f"  HS-UHF converged: {mf_hs.converged}  "
      f"E={mf_hs.e_tot:.6f}  "
      f"time={time.time()-t0:.1f}s")

# ── STEP 2: BROKEN-SYMMETRY UHF ───────────────────────────────
print("Step 2: Broken-symmetry UHF (S=0, antiferromagnetic)...")
mol_bs        = mol_hs.copy()
mol_bs.spin   = 0
mol_bs.build()

t1    = time.time()
mf_bs = scf.UHF(mol_bs)
mf_bs.max_cycle = 800
mf_bs.conv_tol  = 1e-8
mf_bs.verbose   = 3

# Broken-symmetry guess: swap alpha/beta on Fe2
# by flipping the density matrix
dm_hs = mf_hs.make_rdm1()

# Identify Fe2 atom index (index 1) AO indices
# and swap alpha/beta density on those AOs
from pyscf import lo
ao_labels = mol_bs.ao_labels()
fe2_aos   = [i for i, lbl in enumerate(ao_labels)
             if 'Fe2' in lbl or lbl.startswith('1 Fe')]

print(f"  Fe2 AO indices: {fe2_aos[:5]}... ({len(fe2_aos)} total)")

dm_a = dm_hs[0].copy()
dm_b = dm_hs[1].copy()

# Swap alpha and beta density on Fe2 centre
dm_a_new = dm_a.copy()
dm_b_new = dm_b.copy()
for i in fe2_aos:
    for j in fe2_aos:
        dm_a_new[i,j] = dm_b[i,j]
        dm_b_new[i,j] = dm_a[i,j]

mf_bs.kernel((dm_a_new, dm_b_new))
t2 = time.time()

s2_bs = mf_bs.spin_square()[0]
print(f"  BS-UHF converged: {mf_bs.converged}")
print(f"  BS-UHF energy:    {mf_bs.e_tot:.6f} Ha")
print(f"  <S²>:             {s2_bs:.4f}")
print(f"  (expected ~5 for BS state, 0 for true singlet)")
print(f"  Time: {t2-t1:.1f}s")

# ── STEP 3: CASCI ON BS ORBITALS ──────────────────────────────
print("\nStep 3: CASCI(18) on broken-symmetry orbitals...")
e_mean    = (mf_bs.mo_energy[0] + mf_bs.mo_energy[1]) / 2
occ_total = mf_bs.mo_occ[0] + mf_bs.mo_occ[1]
occ_idx   = np.where(occ_total > 0.5)[0]
virt_idx  = np.where(occ_total <= 0.5)[0]
window18  = sorted(list(occ_idx[-9:]) + list(virt_idx[:9]))

n_act_e = 18
for n in [18,16,20,14]:
    if (mol_bs.nelectron-n) >= 0 and (mol_bs.nelectron-n) % 2 == 0:
        n_act_e = n
        break

t3    = time.time()
mo_avg = (mf_bs.mo_coeff[0] + mf_bs.mo_coeff[1]) / 2
mc    = mcscf.CASCI(mf_bs, 18, n_act_e)
mc.verbose = 3
mo    = mc.sort_mo(window18, mo_coeff=mo_avg, base=0)
mc.kernel(mo)
t4    = time.time()

casdm1    = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
no_cas, _ = np.linalg.eigh(casdm1)
no_cas    = np.sort(no_cas)[::-1]
eps       = 1e-12
n_clip    = np.clip(no_cas/2, eps, 1-eps)
s_i       = -(n_clip*np.log(n_clip) + (1-n_clip)*np.log(1-n_clip))
frac      = int(np.sum((no_cas > 0.02) & (no_cas < 1.98)))

print(f"CASCI converged: {mc.converged}")
print(f"CASCI energy:    {mc.e_tot:.6f} Ha")
print(f"NOONs: {np.round(no_cas, 4)}")
print(f"s_i:   {np.round(s_i, 4)}")
print(f"Estimated n_active: {frac}")
print(f"CASCI time: {t4-t3:.1f}s")
print(f"Total time: {t4-t0:.1f}s")

# ── EXCHANGE COUPLING ─────────────────────────────────────────
E_hs = mf_hs.e_tot
E_bs = mf_bs.e_tot
J    = (E_bs - E_hs) / (2 * 5 * (5+1))  # Yamaguchi formula approx
print(f"\nHeisenberg coupling J (approx): {J*27211:.2f} meV")
print(f"  (negative = antiferromagnetic, positive = ferromagnetic)")

# Save
os.makedirs(os.path.expanduser(
    '~/activeml/results/fe2s2_test'), exist_ok=True)
json.dump({
    'hs_converged'  : bool(mf_hs.converged),
    'hs_energy'     : float(mf_hs.e_tot),
    'bs_converged'  : bool(mf_bs.converged),
    'bs_energy'     : float(mf_bs.e_tot),
    'bs_spin_contam': float(s2_bs),
    'casci_converged': bool(mc.converged),
    'casci_energy'  : float(mc.e_tot),
    'no_occ'        : no_cas.tolist(),
    's_i'           : s_i.tolist(),
    'n_active_est'  : frac,
    'J_meV'         : float(J*27211),
    'time_total'    : round(t4-t0, 1),
}, open(os.path.expanduser(
    '~/activeml/results/fe2s2_test/Fe2S2_bs_test.json'),'w'),
   indent=2)
print("Saved.")
