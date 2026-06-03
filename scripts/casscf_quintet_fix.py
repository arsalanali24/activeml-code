"""
Fix FeCl4^2- quintet CASSCF by manually selecting active orbitals.
The problem: default orbital selection picks wrong orbitals for high spin.
The fix: identify the 4 singly-occupied d-orbitals from HF and
         build the active space around them explicitly.
"""
import numpy as np
import json, os
from pyscf import gto, scf, mcscf
from pyscf.tools import molden

scratch = '/scratch/hpc-prf-qehpc/hpcmual/casscf_fecl4_v2'
os.makedirs(scratch, exist_ok=True)

# Build molecule
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
mol.spin    = 4
mol.verbose = 3
mol.build()

# HF
mf = scf.UHF(mol)
mf.max_cycle = 300
mf.conv_tol  = 1e-10
mf.run()
print(f"HF energy: {mf.e_tot:.10f}, converged: {mf.converged}")

# Find singly occupied orbitals from UHF
# alpha - beta occupation difference identifies unpaired electrons
mo_occ_a = mf.mo_occ[0]
mo_occ_b = mf.mo_occ[1]
mo_e_a   = mf.mo_energy[0]
mo_e_b   = mf.mo_energy[1]

print("\nAlpha orbital occupations near Fermi level:")
for i, (e, o) in enumerate(zip(mo_e_a, mo_occ_a)):
    if -1.0 < e < 1.0:
        beta_occ = mo_occ_b[i]
        spin_char = "SINGLY OCC" if o > 0.5 and beta_occ < 0.5 else ""
        print(f"  MO {i+1:3d}: e={e:8.4f}  occ_a={o:.1f}  "
              f"occ_b={beta_occ:.1f}  {spin_char}")

# Build active space: find HOMO-5 to LUMO+4 region
# This captures d-orbitals and nearby ligand orbitals
nmo   = mol.nao
nelec = mol.nelectron
n_occ_a = int(mo_occ_a.sum())
n_occ_b = int(mo_occ_b.sum())

print(f"\nTotal electrons: {nelec}")
print(f"Alpha occupied: {n_occ_a}, Beta occupied: {n_occ_b}")
print(f"Total MOs: {nmo}")

# Select active space: 5 orbitals below HOMO_beta to 5 above LUMO_alpha
# This ensures all 4 singly-occupied d-orbitals are included
homo_b = n_occ_b - 1   # index of highest occupied beta orbital
lumo_a = n_occ_a       # index of lowest unoccupied alpha orbital

# Active space window: HOMO_beta - 4 to LUMO_alpha + 5
start = max(0, homo_b - 4)
end   = min(nmo, lumo_a + 6)
active_mos = list(range(start, end))

print(f"\nSelected active orbital indices: {start} to {end-1}")
print(f"Number of orbitals in window: {len(active_mos)}")

# Use exactly 10 orbitals centered on the d-orbital region
# Take the window and trim to 10 if needed
if len(active_mos) > 10:
    # Center the window on HOMO/LUMO boundary
    center = (homo_b + lumo_a) // 2
    active_mos = list(range(center - 5, center + 5))
    print(f"Trimmed to 10 orbitals: {active_mos[0]} to {active_mos[-1]}")

# CASSCF with explicit orbital selection
mc = mcscf.CASSCF(mf, 10, 10)
mc.max_cycle_macro = 300
mc.max_cycle_micro = 20
mc.conv_tol        = 1e-8
mc.conv_tol_grad   = 1e-4
mc.ah_level_shift  = 1e-3

# Sort MOs to put active orbitals in the right positions
mo = mc.sort_mo(active_mos, base=0)
mc.verbose = 4
mc.kernel(mo)

print(f"\nQuintet CASSCF energy: {mc.e_tot:.10f}")
print(f"Correlation energy:    {mc.e_tot - mf.e_tot:.8f} Ha")
print(f"Converged:             {mc.converged}")
print(f"Iterations:            {mc.iterations}")

if mc.e_tot < mf.e_tot:
    print("PHYSICAL: CASSCF energy below HF ✓")
else:
    print("UNPHYSICAL: CASSCF above HF — wrong active space")

# Natural orbital occupations
casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
no_occ, _ = np.linalg.eigh(casdm1)
no_occ = np.sort(no_occ)[::-1]

print("\nNatural orbital occupations:")
for i, n in enumerate(no_occ):
    flag = " <- ACTIVE" if 0.02 < n < 1.98 else ""
    print(f"  NO {i+1:2d}: {n:.6f}{flag}")

n_active = sum(1 for n in no_occ if 0.02 < n < 1.98)
print(f"\nActive orbitals: {n_active}")

# Save
result = {
    "spin"       : 4,
    "mult"       : 5,
    "E_HF"       : float(mf.e_tot),
    "E_CASSCF"   : float(mc.e_tot),
    "corr_energy": float(mc.e_tot - mf.e_tot),
    "converged"  : bool(mc.converged),
    "no_occ"     : [float(n) for n in no_occ],
    "n_active"   : n_active,
}

outfile = os.path.join(scratch, 'fecl4_quintet_fixed.json')
with open(outfile, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {outfile}")
