"""
HF feature extraction for mixed ligand systems.
Handles: Cl3N1, Cl4O2, Cl2F4 geometries.
Called with: python pyscf_features_mixed.py <casscf_json_path>
"""
import sys, os, json
import numpy as np
import scipy.linalg as la
from pyscf import gto, scf, mp

METAL_Z = {'Fe':26,'Mn':25,'Cr':24,'Co':27,'Ni':28,'Cu':29}

def build_mixed(metal, mix_type, dists):
    import math
    if mix_type == 'Cl3N1':
        d_cl = dists['Cl']; d_n = dists['N']
        s = f"{metal}  0.000  0.000  0.000\n"
        for i in range(3):
            a = i*2*math.pi/3
            s += (f"Cl  {d_cl*math.cos(a):.3f}  "
                  f"{d_cl*math.sin(a):.3f}  0.000\n")
        s += f"N  0.000  0.000  {d_n:.3f}\n"
        return s
    elif mix_type == 'Cl4O2':
        d_cl = dists['Cl']; d_o = dists['O']
        s = f"{metal}  0.000  0.000  0.000\n"
        for p in [(d_cl,0,0),(-d_cl,0,0),
                  (0,d_cl,0),(0,-d_cl,0)]:
            s += f"Cl  {p[0]:.3f}  {p[1]:.3f}  0.000\n"
        s += f"O  0.000  0.000  {d_o:.3f}\n"
        s += f"O  0.000  0.000  {-d_o:.3f}\n"
        return s
    elif mix_type == 'Cl2F4':
        d_cl = dists['Cl']; d_f = dists['F']
        s = f"{metal}  0.000  0.000  0.000\n"
        for p in [(d_f,0,0),(-d_f,0,0),
                  (0,d_f,0),(0,-d_f,0)]:
            s += f"F  {p[0]:.3f}  {p[1]:.3f}  0.000\n"
        s += f"Cl  0.000  0.000  {d_cl:.3f}\n"
        s += f"Cl  0.000  0.000  {-d_cl:.3f}\n"
        return s
    raise ValueError(f"Unknown: {mix_type}")

# Equilibrium distances
EQ = {
    'Fe':{'Cl':2.18,'F':1.85,'N':2.10,'O':2.05},
    'Mn':{'Cl':2.35,'F':1.98,'N':2.20,'O':2.15},
    'Cr':{'Cl':2.31,'F':1.94,'N':2.10,'O':2.05},
    'Co':{'Cl':2.26,'F':1.90,'N':2.00,'O':1.95},
    'Ni':{'Cl':2.21,'F':1.86,'N':2.05,'O':2.00},
    'Cu':{'Cl':2.26,'F':1.91,'N':2.05,'O':1.98},
}

# Read CASSCF json to get system info
cas_path = sys.argv[1]
cas = json.load(open(cas_path))

metal    = cas['metal']
mix_type = cas['ligand']
charge   = cas['charge']
spin     = cas['spin']
frac     = cas.get('bond_frac', 1.0)
name     = cas['name']

outdir  = os.path.expanduser('~/activeml/data/generated')
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{name}.json")

if os.path.exists(outfile):
    print(f"EXISTS: {outfile}")
    sys.exit(0)

# Build geometry
dists = {lig: round(EQ[metal][lig]*frac, 3)
         for lig in EQ[metal]}

mol = gto.Mole()
mol.atom    = build_mixed(metal, mix_type, dists)
mol.basis   = 'def2-SVP'
mol.charge  = charge
mol.spin    = spin
mol.verbose = 0
mol.build()

# UHF
mf = scf.UHF(mol)
mf.max_cycle = 500
mf.conv_tol  = 1e-9
mf.verbose   = 0
mf.run()

s_ideal     = spin/2.0
spin_contam = float(abs(mf.spin_square()[0] -
                        s_ideal*(s_ideal+1)))

# Orbital features
S_ovlp = mol.intor('int1e_ovlp')
X_orth = la.inv(la.sqrtm(S_ovlp))
dm_a, dm_b = mf.make_rdm1()
dm_orth = X_orth @ (dm_a+dm_b) @ X_orth.T
uno_occ = np.sort(np.linalg.eigvalsh(dm_orth))[::-1]
n_frac_uno_010 = int(np.sum((uno_occ>0.10)&(uno_occ<1.90)))

e_a = mf.mo_energy[0]; e_b = mf.mo_energy[1]
occ_a = mf.mo_occ[0];  occ_b = mf.mo_occ[1]
e_mean    = (e_a+e_b)/2
occ_total = occ_a+occ_b
homo_e = float(e_mean[occ_total>0.5].max())
lumo_e = float(e_mean[occ_total<0.5].min())
gap_cen = (homo_e+lumo_e)/2
d_region = [float(e_mean[i]) for i in range(len(e_mean))
            if abs(e_mean[i]-gap_cen)<2.0]
d_bw = float(max(d_region)-min(d_region)) \
       if len(d_region)>1 else 0.0

energies = e_mean
occs     = occ_total
n_orbs   = len(energies)
n_w01 = int(np.sum(np.abs(energies-gap_cen)<0.1))
n_w03 = int(np.sum(np.abs(energies-gap_cen)<0.3))
n_w05 = int(np.sum(np.abs(energies-gap_cen)<0.5))
near_e = np.sort(energies[np.abs(energies-gap_cen)<1.0])
if len(near_e)>1:
    g = np.diff(near_e)
    max_gn = float(g.max()); mean_gn = float(g.mean())
else:
    max_gn = mean_gn = 0.0

n_sing = int(np.sum((occs>0.5)&(occs<1.5)))
occ_e  = np.sort(energies[occs>1.5])[::-1]
occ_spread = float(occ_e[0]-occ_e[-1]) if len(occ_e)>1 else 0.0
h1g = float(occ_e[0]-occ_e[1]) if len(occ_e)>1 else 0.0
h2g = float(occ_e[0]-occ_e[2]) if len(occ_e)>2 else 0.0

homo_a = float(e_a[occ_a>0.5].max()) if (occ_a>0.5).any() else 0.0
homo_b = float(e_b[occ_b>0.5].max()) if (occ_b>0.5).any() else 0.0
homo_ab_gap = float(abs(homo_a-homo_b))

# delta_E_HS_LS
delta_E = 0.0
hs_spin = min(mol.nelectron, 8)
if (hs_spin%2)!=(mol.nelectron%2): hs_spin -= 1
if hs_spin > spin:
    try:
        mol_hs = gto.Mole()
        mol_hs.atom    = build_mixed(metal, mix_type, dists)
        mol_hs.basis   = 'def2-SVP'
        mol_hs.charge  = charge
        mol_hs.spin    = hs_spin
        mol_hs.verbose = 0
        mol_hs.build()
        mf_hs = scf.UHF(mol_hs)
        mf_hs.max_cycle=300; mf_hs.conv_tol=1e-8
        mf_hs.verbose=0; mf_hs.run()
        if mf_hs.converged:
            delta_E = float(mf.e_tot - mf_hs.e_tot)
    except: pass

# MP2
mp2_corr = 0.0; largest_t2 = 0.0; mi_entropy = 0.0
try:
    mp2 = mp.MP2(mf); mp2.verbose=0; mp2.run()
    mp2_corr = float(mp2.e_corr)
    t2 = mp2.t2
    if isinstance(t2, tuple):
        largest_t2 = float(max(np.abs(t).max()
                               for t in t2 if t is not None))
    eps = 1e-12
    n_clip = np.clip(uno_occ/2, eps, 1-eps)
    mi_entropy = float(-(n_clip*np.log(n_clip)+
                         (1-n_clip)*np.log(1-n_clip)).sum())
except: pass

# Build orbital list
mol_feats = {
    'spin_contam'     : spin_contam,
    'd_bandwidth'     : d_bw,
    'multiplicity'    : spin+1,
    'n_frac_uno_010'  : n_frac_uno_010,
    'delta_E_HS_LS'   : delta_E,
    'homo_ab_gap'     : homo_ab_gap,
    'mp2_corr'        : mp2_corr,
    'largest_t2'      : largest_t2,
    'mi_entropy_sum'  : mi_entropy,
    'd_electron_count': -1,
    'spec_strength'   : 4,
    'coord_number'    : cas.get('n_ligands', 6),
}

orbitals = []
for i in range(n_orbs):
    e_i = float(energies[i]); occ_i = float(occs[i])
    gaps = [abs(e_i-float(energies[j]))
            for j in range(n_orbs) if j!=i]
    near_gap = float(min(gaps))
    ct = [abs(e_i-float(energies[j]))
          for j in range(n_orbs)
          if j!=i and (occs[j]<0.5 if occ_i>0.5
                       else occs[j]>0.5)]
    ct_gap = float(min(ct)) if ct else 999.0
    occ_dev = float(min(occ_i, 2.0-occ_i))
    orb = {
        'index'          : i,
        'energy'         : e_i,
        'occupation'     : occ_i,
        'uno_occ'        : occ_i,
        'uno_frac_dev'   : occ_dev,
        'dist_homo'      : float(abs(e_i-homo_e)),
        'dist_lumo'      : float(abs(e_i-lumo_e)),
        'dist_gap_center': float(abs(e_i-gap_cen)),
        'nearest_gap'    : near_gap,
        'ct_gap'         : ct_gap,
        'n_near_orbs'    : sum(1 for j in range(n_orbs)
                               if j!=i and
                               abs(e_i-float(energies[j]))<0.3),
    }
    orb.update(mol_feats)
    orbitals.append(orb)

record = {
    'name'          : name,
    'metal'         : metal,
    'ligand'        : mix_type,
    'n_ligands'     : cas.get('n_ligands',6),
    'charge'        : charge,
    'spin'          : spin,
    'mult'          : spin+1,
    'geometry'      : 'mixed',
    'n_electrons'   : mol.nelectron,
    'E_HF'          : float(mf.e_tot),
    'homo_energy'   : homo_e,
    'lumo_energy'   : lumo_e,
    'homo_lumo_gap' : lumo_e-homo_e,
    'n_within_01'   : n_w01,
    'n_within_03'   : n_w03,
    'n_within_05'   : n_w05,
    'max_gap_near'  : max_gn,
    'mean_gap_near' : mean_gn,
    'n_singly_occ'  : n_sing,
    'occ_spread'    : occ_spread,
    'homo_1_gap'    : h1g,
    'homo_2_gap'    : h2g,
    'orbitals'      : orbitals,
}
record.update(mol_feats)

with open(outfile,'w') as f:
    json.dump(record, f, indent=2)
print(f"Saved: {outfile}")
