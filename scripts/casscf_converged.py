"""
FeCl4^2- CASSCF at three spin states with proper convergence.
Fixed: removed project_init_guess which fails for UHF.
"""
import numpy as np
import json, os
from pyscf import gto, scf, mcscf

scratch = '/scratch/hpc-prf-qehpc/hpcmual/casscf_fecl4_v2'
os.makedirs(scratch, exist_ok=True)

def build_mol(spin):
    mol = gto.Mole()
    mol.atom = '''
Fe   0.000   0.000   0.000
Cl   2.180   0.000   0.000
Cl  -2.180   0.000   0.000
Cl   0.000   2.180   0.000
Cl   0.000  -2.180   0.000
'''
    mol.basis  = 'def2-SVP'
    mol.charge = -2
    mol.spin   = spin
    mol.verbose = 0
    mol.build()
    return mol

results = {}

for spin in [4, 2, 0]:
    mult = spin + 1
    print(f"\n{'='*55}")
    print(f"FeCl4^2-  spin={spin}  multiplicity={mult}")
    print(f"{'='*55}")

    mol = build_mol(spin)

    # HF
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol  = 1e-10
    mf.run()
    print(f"HF energy:  {mf.e_tot:.10f}  converged: {mf.converged}")

    # CASSCF — convergence fixes only, no project_init_guess
    mc = mcscf.CASSCF(mf, 10, 10)
    mc.max_cycle_macro = 200
    mc.max_cycle_micro = 20
    mc.conv_tol        = 1e-9
    mc.conv_tol_grad   = 1e-5
    mc.ah_level_shift  = 1e-4
    mc.verbose         = 3
    mc.run()

    print(f"CASSCF energy:  {mc.e_tot:.10f}")
    print(f"Correlation:    {mc.e_tot - mf.e_tot:.8f} Ha")
    print(f"Converged:      {mc.converged}")

    # Natural orbital occupations
    casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    no_occ, _ = np.linalg.eigh(casdm1)
    no_occ = np.sort(no_occ)[::-1]

    print(f"\nNatural orbital occupations:")
    for i, n in enumerate(no_occ):
        flag = " <- ACTIVE" if 0.02 < n < 1.98 else ""
        print(f"  NO {i+1:2d}: {n:.6f}{flag}")

    n_active = sum(1 for n in no_occ if 0.02 < n < 1.98)
    print(f"\nActive orbitals: {n_active}")

    results[f"mult_{mult}"] = {
        "spin"       : spin,
        "mult"       : mult,
        "E_HF"       : float(mf.e_tot),
        "E_CASSCF"   : float(mc.e_tot),
        "corr_energy": float(mc.e_tot - mf.e_tot),
        "converged"  : bool(mc.converged),
        "no_occ"     : [float(n) for n in no_occ],
        "n_active"   : n_active,
    }

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Mult':>5}  {'E_CASSCF':>18}  {'Ecorr':>10}  "
      f"{'NActive':>8}  {'Conv':>6}")
print("-"*55)
for key, r in results.items():
    print(f"  {r['mult']:>4}  {r['E_CASSCF']:>18.10f}  "
          f"{r['corr_energy']:>10.6f}  "
          f"{r['n_active']:>8}  "
          f"{'YES' if r['converged'] else 'NO':>6}")

outfile = os.path.join(scratch, 'fecl4_converged.json')
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)

home_out = os.path.expanduser(
    '~/activeml/data/generated/fecl4_converged.json')
with open(home_out, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {outfile}")
print(f"Saved: {home_out}")
