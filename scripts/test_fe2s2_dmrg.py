import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_INTERFACE_LAYER"] = "LP64"
"""
Test DMRG(20,20) on Fe2S2(SH)4 2-
Broken-symmetry reference, bond dim 500
"""
import numpy as np
from pyscf import gto, scf, mcscf
from pyblock2.dmrgscf import DMRGCI
import json, time, os

# ── MOLECULE ──────────────────────────────────────────────────
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
mol.spin    = 0       # antiferromagnetic singlet
mol.basis   = 'def2-SVP'
mol.verbose = 3
mol.build()

print(f"System: Fe2S2(SH)4 2-")
print(f"Electrons: {mol.nelectron}")
print(f"Basis functions: {mol.nao}")

# ── STEP 1: HIGH-SPIN UHF ─────────────────────────────────────
print("\n--- Step 1: High-spin UHF ---")
mol_hs      = mol.copy()
mol_hs.spin = 10
mol_hs.build()
t0  = time.time()
mf_hs = scf.UHF(mol_hs)
mf_hs.max_cycle = 500
mf_hs.conv_tol  = 1e-9
mf_hs.verbose   = 0
mf_hs.kernel()
print(f"HS-UHF: converged={mf_hs.converged} "
      f"E={mf_hs.e_tot:.6f} time={time.time()-t0:.1f}s")

# ── STEP 2: BROKEN-SYMMETRY UHF ───────────────────────────────
print("\n--- Step 2: Broken-symmetry UHF ---")
t1    = time.time()
mf_bs = scf.UHF(mol)
mf_bs.max_cycle = 800
mf_bs.conv_tol  = 1e-8
mf_bs.verbose   = 0

# BS guess: swap alpha/beta on Fe2
dm_hs = mf_hs.make_rdm1()
ao_labels = mol.ao_labels()
fe2_aos   = [i for i,l in enumerate(ao_labels)
             if l.split()[0] == '1']
dm_a = dm_hs[0].copy()
dm_b = dm_hs[1].copy()
dm_a_new = dm_a.copy()
dm_b_new = dm_b.copy()
for i in fe2_aos:
    for j in fe2_aos:
        dm_a_new[i,j] = dm_b[i,j]
        dm_b_new[i,j] = dm_a[i,j]
mf_bs.kernel((dm_a_new, dm_b_new))
s2 = mf_bs.spin_square()[0]
print(f"BS-UHF: converged={mf_bs.converged} "
      f"E={mf_bs.e_tot:.6f} S2={s2:.3f} "
      f"time={time.time()-t1:.1f}s")

# ── STEP 3: BUILD 20-ORBITAL WINDOW ───────────────────────────
print("\n--- Step 3: DMRG(20, n_e) ---")
e_mean    = (mf_bs.mo_energy[0] + mf_bs.mo_energy[1]) / 2
occ_total = mf_bs.mo_occ[0] + mf_bs.mo_occ[1]
occ_idx   = np.where(occ_total > 0.5)[0]
virt_idx  = np.where(occ_total <= 0.5)[0]
window20  = sorted(list(occ_idx[-10:]) + list(virt_idx[:10]))
print(f"Window: {window20}")

# Active electrons in window
n_act_e = 20
for n in [20, 18, 22, 16, 24]:
    if (mol.nelectron - n) >= 0 and (mol.nelectron - n) % 2 == 0:
        n_act_e = n
        break
print(f"Active electrons: {n_act_e}")

# ── STEP 4: DMRG CALCULATION ──────────────────────────────────
t2     = time.time()
mo_avg = (mf_bs.mo_coeff[0] + mf_bs.mo_coeff[1]) / 2

mc = mcscf.CASCI(mf_bs, 20, n_act_e)
mc.fcisolver = DMRGCI(mf_bs)
mc.fcisolver.dmrg_args = {"startM": 100, "maxM": 200, "schedule": "default", "sweep_tol": 1e-6, "memory": 80000000000, "cutoff": 1e-14}    # bond dimension
mc.max_cycle_macro     = 20
mc.max_cycle_micro     = 10
mc.conv_tol            = 1e-7
mc.verbose             = 3

mo = mc.sort_mo(window20, mo_coeff=mo_avg, base=0)
mc.kernel(mo)
t3 = time.time()

# ── STEP 5: EXTRACT NOONs AND ENTROPY ─────────────────────────
casdm1    = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
no_cas, _ = np.linalg.eigh(casdm1)
no_cas    = np.sort(no_cas)[::-1]
eps       = 1e-12
n_clip    = np.clip(no_cas/2, eps, 1-eps)
s_i       = -(n_clip*np.log(n_clip) +
              (1-n_clip)*np.log(1-n_clip))
frac      = int(np.sum((no_cas > 0.02) & (no_cas < 1.98)))

print(f"\nDMRG converged: {mc.converged}")
print(f"DMRG energy:    {mc.e_tot:.6f} Ha")
print(f"Time: {t3-t2:.1f}s")
print(f"\nNOONs: {np.round(no_cas, 4)}")
print(f"s_i:   {np.round(s_i, 4)}")
print(f"n_active estimate: {frac}")

# Exchange coupling
E_hs = mf_hs.e_tot
E_bs = mf_bs.e_tot
J    = (E_bs - E_hs) / (2 * 5 * 6) * 27211
print(f"\nJ coupling: {J:.1f} meV "
      f"({'antiferro' if J<0 else 'ferro'}magnetic)")

# Save
os.makedirs(os.path.expanduser(
    '~/activeml/results/fe2s2_test'), exist_ok=True)
json.dump({
    'system'         : 'Fe2S2SH4_2m_BS',
    'n_electrons'    : mol.nelectron,
    'n_basis'        : mol.nao,
    'bs_converged'   : bool(mf_bs.converged),
    'bs_energy'      : float(mf_bs.e_tot),
    'bs_spin_contam' : float(s2),
    'dmrg_converged' : bool(mc.converged),
    'dmrg_energy'    : float(mc.e_tot),
    'no_occ'         : no_cas.tolist(),
    's_i'            : s_i.tolist(),
    'n_active'       : frac,
    'J_meV'          : float(J),
    'time_dmrg'      : round(t3-t2, 1),
    'time_total'     : round(t3-t0, 1),
}, open(os.path.expanduser(
    '~/activeml/results/fe2s2_test/Fe2S2_dmrg.json'),'w'),
   indent=2)
print("\nSaved.")
