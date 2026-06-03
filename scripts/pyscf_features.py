"""
Extract HF features for one TM complex.
Includes physics-informed features:
  - d_electron_count, spin_contamination,
    spectrochemical_strength, coord_number, d_bandwidth
Usage: python pyscf_features.py Metal Ligand N_lig Charge Spin [Dist]
"""
import sys, os, json
import numpy as np
from pyscf import gto, scf

# ── LOOKUP TABLES ─────────────────────────────────────────────
D_ELECTRONS = {
    'Cr':{2:4,3:3,4:2,5:1,6:0},
    'Mn':{2:5,3:4,4:3,5:2,6:1,7:0},
    'Fe':{2:6,3:5,4:4,5:3,6:2},
    'Co':{2:7,3:6,4:5},
    'Ni':{2:8,3:7,4:6},
    'Cu':{1:10,2:9,3:8},
    'Zn':{2:10},
    'Ti':{2:2,3:1,4:0},
    'V' :{2:3,3:2,4:1,5:0},
    'Sc':{3:0},
}

SPEC_STRENGTH = {
    'I':1,'Br':2,'Cl':3,'S':4,'F':5,
    'O':6,'N':7,'P':8,'C':9,'CO':10,'CN':11,
}

LIG_CHARGE = {
    'Cl':-1,'F':-1,'Br':-1,'I':-1,
    'O':0,'N':0,'S':-2,'C':0,
}

def get_ox_state(metal, charge, n_lig, ligand):
    lc = LIG_CHARGE.get(ligand, 0)
    return charge - (n_lig * lc)

def build_geometry(metal, ligand, n_lig, dist):
    positions = {
        4: [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6: [(dist,0,0),(-dist,0,0),(0,dist,0),
            (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[n_lig]:
        atom_str += (f"{ligand}  {p[0]:.3f}  "
                     f"{p[1]:.3f}  {p[2]:.3f}\n")
    return atom_str

# ── ARGUMENTS ─────────────────────────────────────────────────
metal   = sys.argv[1]
ligand  = sys.argv[2]
n_lig   = int(sys.argv[3])
charge  = int(sys.argv[4])
spin    = int(sys.argv[5])
dist    = float(sys.argv[6]) if len(sys.argv) > 6 else 2.18

name    = f"{metal}_{ligand}{n_lig}_chg{charge}_spin{spin}"
outdir  = os.path.expanduser("~/activeml/data/generated")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{name}.json")

# ── BUILD MOLECULE ────────────────────────────────────────────
mol = gto.Mole()
mol.atom    = build_geometry(metal, ligand, n_lig, dist)
mol.basis   = 'def2-SVP'
mol.charge  = charge
mol.spin    = spin
mol.verbose = 3
mol.build()

print(f"System: {name}")
print(f"Electrons: {mol.nelectron}  "
      f"Basis: {mol.nao}  Mult: {spin+1}")

# ── HF ────────────────────────────────────────────────────────
mf = scf.UHF(mol)
mf.max_cycle = 300
mf.conv_tol  = 1e-10
mf.run()
print(f"HF energy: {mf.e_tot:.8f}  conv: {mf.converged}")

# ── PHYSICS FEATURES (molecule-level) ────────────────────────
ox_state    = get_ox_state(metal, charge, n_lig, ligand)
d_elec      = D_ELECTRONS.get(metal, {}).get(ox_state, -1)
spec_str    = SPEC_STRENGTH.get(ligand, 5)
coord_num   = n_lig

s_ideal     = spin / 2.0
s2_ideal    = s_ideal * (s_ideal + 1)
s2_actual   = mf.spin_square()[0]
spin_contam = float(abs(s2_actual - s2_ideal))

print(f"Oxidation: {metal}{ox_state:+d}  "
      f"d-electrons: {d_elec}  "
      f"spin_contam: {spin_contam:.4f}")

# ── ORBITAL FEATURES ─────────────────────────────────────────
e_a   = mf.mo_energy[0]
e_b   = mf.mo_energy[1]
occ_a = mf.mo_occ[0]
occ_b = mf.mo_occ[1]

n_orbs    = len(e_a)
e_mean    = (e_a + e_b) / 2
occ_total = occ_a + occ_b

occ_mask  = occ_total > 0.5
virt_mask = occ_total < 0.5
homo_e    = float(e_mean[occ_mask].max()) \
            if occ_mask.any() else 0.0
lumo_e    = float(e_mean[virt_mask].min()) \
            if virt_mask.any() else 0.0
gap_cen   = (homo_e + lumo_e) / 2

# d-bandwidth: spread of orbitals within 2 Ha of Fermi level
d_region    = [float(e_mean[i]) for i in range(n_orbs)
               if abs(e_mean[i] - gap_cen) < 2.0]
d_bandwidth = float(max(d_region) - min(d_region)) \
              if len(d_region) > 1 else 0.0

orbitals = []
for i in range(n_orbs):
    e_i   = float(e_mean[i])
    occ_i = float(occ_total[i])

    gaps  = [abs(e_i - float(e_mean[j]))
             for j in range(n_orbs) if j != i]
    near_gap = float(min(gaps))

    if occ_i > 0.5:
        ct = [abs(e_i - float(e_mean[j]))
              for j in range(n_orbs)
              if j != i and occ_total[j] < 0.5]
    else:
        ct = [abs(e_i - float(e_mean[j]))
              for j in range(n_orbs)
              if j != i and occ_total[j] > 0.5]
    ct_gap = float(min(ct)) if ct else 999.0

    n_near = sum(1 for j in range(n_orbs)
                 if j != i and
                 abs(e_i - float(e_mean[j])) < 0.3)

    orbitals.append({
        "index"           : i,
        "energy"          : e_i,
        "occupation"      : occ_i,
        "dist_homo"       : float(abs(e_i - homo_e)),
        "dist_lumo"       : float(abs(e_i - lumo_e)),
        "dist_gap_center" : float(abs(e_i - gap_cen)),
        "nearest_gap"     : near_gap,
        "ct_gap"          : ct_gap,
        "n_near_orbs"     : n_near,
        # New physics features
        "d_electron_count": d_elec,
        "spin_contam"     : spin_contam,
        "spec_strength"   : spec_str,
        "coord_number"    : coord_num,
        "d_bandwidth"     : d_bandwidth,
        "multiplicity"    : spin + 1,
    })

record = {
    "name"              : name,
    "metal"             : metal,
    "ligand"            : ligand,
    "n_ligands"         : n_lig,
    "charge"            : charge,
    "spin"              : spin,
    "mult"              : spin + 1,
    "dist_ang"          : dist,
    "n_electrons"       : mol.nelectron,
    "E_HF"              : float(mf.e_tot),
    "homo_energy"       : homo_e,
    "lumo_energy"       : lumo_e,
    "homo_lumo_gap"     : lumo_e - homo_e,
    "d_electron_count"  : d_elec,
    "spin_contamination": spin_contam,
    "spec_strength"     : spec_str,
    "coord_number"      : coord_num,
    "d_bandwidth"       : d_bandwidth,
    "orbitals"          : orbitals,
}

with open(outfile, "w") as f:
    json.dump(record, f, indent=2)

print(f"Saved: {outfile}")
print(f"Orbitals saved: {len(orbitals)}")
