"""
CASSCF for FeCl4^2- at three spin states.
Prints natural orbital occupations clearly.
"""
import numpy as np
import json, os
from pyscf import gto, scf, mcscf

scratch = '/scratch/hpc-prf-qehpc/hpcmual/casscf_fecl4'
os.makedirs(scratch, exist_ok=True)

results = {}

for spin in [4, 2, 0]:   # quintet, triplet, singlet
    mult = spin + 1
    print(f"\n{'='*50}")
    print(f"FeCl4^2-  spin={spin}  mult={mult}")
    print(f"{'='*50}")

    mol = gto.Mole()
    mol.atom = '''
Fe   0.000   0.000   0.000
Cl   2.180   0.000   0.000
Cl  -2.180   0.000   0.000
Cl   0.000   2.180   0.000
Cl   0.000  -2.180   0.000
'''
    mol.basis   = 'def2-SVP'
    mol.charge  = -2
    mol.spin    = spin
    mol.verbose = 0
    mol.build()

    # HF
    mf = scf.UHF(mol)
    mf.run()
    print(f"HF energy:     {mf.e_tot:.8f}")

    # CASSCF — 10 electrons in 10 orbitals
    mc = mcscf.CASSCF(mf, 10, 10)
    mc.verbose = 0
    mc.run()
    print(f"CASSCF energy: {mc.e_tot:.8f}")
    print(f"Correlation:   {mc.e_tot - mf.e_tot:.8f} Ha")
    print(f"Converged:     {mc.converged}")

    # Natural orbital occupations of active space
    # mc.mo_occ gives occupations of ALL MOs
    # active space is mc.ncore to mc.ncore+mc.ncas
    ncore = mc.ncore
    ncas  = mc.ncas

    # Get natural orbital occupations properly
    casdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, mc.nelecas)
    no_occ, _ = np.linalg.eigh(casdm1)
    no_occ = np.sort(no_occ)[::-1]

    print(f"\nActive space natural orbital occupations:")
    for i, n in enumerate(no_occ):
        flag = " <- ACTIVE" if 0.02 < n < 1.98 else ""
        print(f"  NO {i+1:2d}: {n:.6f}{flag}")

    n_active = sum(1 for n in no_occ if 0.02 < n < 1.98)
    print(f"\nActive orbitals (0.02 < occ < 1.98): {n_active}")

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

# Summary
print("\n" + "="*60)
print("SUMMARY: Active space comparison across spin states")
print("="*60)
for key, r in results.items():
    print(f"\nMult={r['mult']} (spin={r['spin']}):")
    print(f"  CASSCF energy: {r['E_CASSCF']:.8f}")
    print(f"  Active orbs:   {r['n_active']}")
    print(f"  NO occs: {np.round(r['no_occ'], 4)}")

# Save JSON
outfile = os.path.join(scratch, 'fecl4_three_spins.json')
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outfile}")
