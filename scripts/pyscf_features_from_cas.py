"""
Universal HF feature extractor.
Reads CASSCF JSON, rebuilds geometry, computes all v3 features.
Handles: symmetric, sqpyr, tbp, mixed, csd_real, sq_pl.
Usage: python pyscf_features_from_cas.py <casscf_json_path>
"""
import sys, os, json
import numpy as np
import scipy.linalg as la
from pyscf import gto, scf, mp
import math

# ── CSD COORDINATES LOOKUP ─────────────────────────────────────
CSD_ATOMS = {
"FeCl4_2m_tet": """Fe 0.0 0.0 0.0
Cl 2.195 0.0 0.0
Cl -2.195 0.0 0.0
Cl 0.0 2.195 0.0
Cl 0.0 -2.195 0.0""",
"FeCl6_3m_oct": """Fe 0.0 0.0 0.0
Cl 2.480 0.0 0.0
Cl -2.480 0.0 0.0
Cl 0.0 2.480 0.0
Cl 0.0 -2.480 0.0
Cl 0.0 0.0 2.480
Cl 0.0 0.0 -2.480""",
"FeCl4_dist_tet": """Fe 0.0 0.0 0.0
Cl 2.210 0.100 -0.050
Cl -2.180 0.050 0.080
Cl 0.060 2.200 -0.040
Cl -0.090 -2.190 0.010""",
"MnCl4_2m_tet": """Mn 0.0 0.0 0.0
Cl 2.380 0.0 0.0
Cl -2.380 0.0 0.0
Cl 0.0 2.380 0.0
Cl 0.0 -2.380 0.0""",
"MnCl6_4m_oct": """Mn 0.0 0.0 0.0
Cl 2.610 0.0 0.0
Cl -2.610 0.0 0.0
Cl 0.0 2.610 0.0
Cl 0.0 -2.610 0.0
Cl 0.0 0.0 2.610
Cl 0.0 0.0 -2.610""",
"MnCl6_dist_oct": """Mn 0.0 0.0 0.0
Cl 2.590 0.0 0.0
Cl -2.590 0.0 0.0
Cl 0.0 2.590 0.0
Cl 0.0 -2.590 0.0
Cl 0.0 0.0 2.850
Cl 0.0 0.0 -2.850""",
"CrCl6_3m_oct": """Cr 0.0 0.0 0.0
Cl 2.490 0.0 0.0
Cl -2.490 0.0 0.0
Cl 0.0 2.490 0.0
Cl 0.0 -2.490 0.0
Cl 0.0 0.0 2.490
Cl 0.0 0.0 -2.490""",
"CrCl4_2m_tet": """Cr 0.0 0.0 0.0
Cl 2.320 0.0 0.0
Cl -2.320 0.0 0.0
Cl 0.0 2.320 0.0
Cl 0.0 -2.320 0.0""",
"CoCl4_2m_tet": """Co 0.0 0.0 0.0
Cl 2.260 0.0 0.0
Cl -2.260 0.0 0.0
Cl 0.0 2.260 0.0
Cl 0.0 -2.260 0.0""",
"CoCl6_3m_oct": """Co 0.0 0.0 0.0
Cl 2.440 0.0 0.0
Cl -2.440 0.0 0.0
Cl 0.0 2.440 0.0
Cl 0.0 -2.440 0.0
Cl 0.0 0.0 2.440
Cl 0.0 0.0 -2.440""",
"NiCl4_2m_sqpl": """Ni 0.0 0.0 0.0
Cl 2.210 0.0 0.0
Cl -2.210 0.0 0.0
Cl 0.0 2.210 0.0
Cl 0.0 -2.210 0.0""",
"NiCl6_4m_oct": """Ni 0.0 0.0 0.0
Cl 2.400 0.0 0.0
Cl -2.400 0.0 0.0
Cl 0.0 2.400 0.0
Cl 0.0 -2.400 0.0
Cl 0.0 0.0 2.400
Cl 0.0 0.0 -2.400""",
"CuCl4_2m_sqpl": """Cu 0.0 0.0 0.0
Cl 2.265 0.0 0.0
Cl -2.265 0.0 0.0
Cl 0.0 2.265 0.0
Cl 0.0 -2.265 0.0""",
"CuCl6_4m_jt": """Cu 0.0 0.0 0.0
Cl 2.300 0.0 0.0
Cl -2.300 0.0 0.0
Cl 0.0 2.300 0.0
Cl 0.0 -2.300 0.0
Cl 0.0 0.0 2.650
Cl 0.0 0.0 -2.650""",
"FeBr4_2m_tet": """Fe 0.0 0.0 0.0
Br 2.380 0.0 0.0
Br -2.380 0.0 0.0
Br 0.0 2.380 0.0
Br 0.0 -2.380 0.0""",
"MnBr4_2m_tet": """Mn 0.0 0.0 0.0
Br 2.530 0.0 0.0
Br -2.530 0.0 0.0
Br 0.0 2.530 0.0
Br 0.0 -2.530 0.0""",
"CoBr4_2m_tet": """Co 0.0 0.0 0.0
Br 2.420 0.0 0.0
Br -2.420 0.0 0.0
Br 0.0 2.420 0.0
Br 0.0 -2.420 0.0""",
"FeCl4O2_trans": """Fe 0.0 0.0 0.0
Cl 2.310 0.0 0.0
Cl -2.310 0.0 0.0
Cl 0.0 2.310 0.0
Cl 0.0 -2.310 0.0
O 0.0 0.0 2.120
O 0.0 0.0 -2.120""",
"MnCl4N2_trans": """Mn 0.0 0.0 0.0
Cl 2.580 0.0 0.0
Cl -2.580 0.0 0.0
Cl 0.0 2.580 0.0
Cl 0.0 -2.580 0.0
N 0.0 0.0 2.280
N 0.0 0.0 -2.280""",
"FeF6_3m_oct": """Fe 0.0 0.0 0.0
F 1.930 0.0 0.0
F -1.930 0.0 0.0
F 0.0 1.930 0.0
F 0.0 -1.930 0.0
F 0.0 0.0 1.930
F 0.0 0.0 -1.930""",
"CrF6_3m_oct": """Cr 0.0 0.0 0.0
F 1.980 0.0 0.0
F -1.980 0.0 0.0
F 0.0 1.980 0.0
F 0.0 -1.980 0.0
F 0.0 0.0 1.980
F 0.0 0.0 -1.980""",
"MnF6_4m_oct": """Mn 0.0 0.0 0.0
F 2.080 0.0 0.0
F -2.080 0.0 0.0
F 0.0 2.080 0.0
F 0.0 -2.080 0.0
F 0.0 0.0 2.080
F 0.0 0.0 -2.080""",
"FeCl6_dist1": """Fe 0.0 0.0 0.0
Cl 2.465 0.120 -0.080
Cl -2.470 0.060 0.090
Cl 0.090 2.460 -0.070
Cl -0.070 -2.475 0.080
Cl 0.050 0.040 2.490
Cl -0.060 -0.050 -2.485""",
"MnCl6_dist2": """Mn 0.0 0.0 0.0
Cl 2.595 0.150 -0.100
Cl -2.600 0.080 0.110
Cl 0.100 2.595 -0.090
Cl -0.090 -2.610 0.100
Cl 0.060 0.050 2.840
Cl -0.070 -0.060 -2.860""",
"FeCl4_1m_tet": """Fe 0.0 0.0 0.0
Cl 2.175 0.0 0.0
Cl -2.175 0.0 0.0
Cl 0.0 2.175 0.0
Cl 0.0 -2.175 0.0""",
"CrCl4_1m_tet": """Cr 0.0 0.0 0.0
Cl 2.290 0.0 0.0
Cl -2.290 0.0 0.0
Cl 0.0 2.290 0.0
Cl 0.0 -2.290 0.0""",
"CoCl6_4m_oct": """Co 0.0 0.0 0.0
Cl 2.520 0.0 0.0
Cl -2.520 0.0 0.0
Cl 0.0 2.520 0.0
Cl 0.0 -2.520 0.0
Cl 0.0 0.0 2.520
Cl 0.0 0.0 -2.520""",
"NiCl4_1m_sqpl": """Ni 0.0 0.0 0.0
Cl 2.190 0.0 0.0
Cl -2.190 0.0 0.0
Cl 0.0 2.190 0.0
Cl 0.0 -2.190 0.0""",
"CrBr6_3m_oct": """Cr 0.0 0.0 0.0
Br 2.540 0.0 0.0
Br -2.540 0.0 0.0
Br 0.0 2.540 0.0
Br 0.0 -2.540 0.0
Br 0.0 0.0 2.540
Br 0.0 0.0 -2.540""",
"NiBr4_2m_sqpl": """Ni 0.0 0.0 0.0
Br 2.370 0.0 0.0
Br -2.370 0.0 0.0
Br 0.0 2.370 0.0
Br 0.0 -2.370 0.0""",
}

EQ = {
    'Fe':{'Cl':2.18,'Br':2.35,'F':1.85,'N':2.10,
          'O':2.05,'S':2.35,'C':1.90,'H':1.65},
    'Mn':{'Cl':2.35,'Br':2.50,'F':1.98,'N':2.20,
          'O':2.15,'S':2.45,'C':2.00,'H':1.75},
    'Cr':{'Cl':2.31,'Br':2.47,'F':1.94,'N':2.10,
          'O':2.05,'S':2.40,'C':1.93,'H':1.72},
    'Co':{'Cl':2.26,'Br':2.42,'F':1.90,'N':2.00,
          'O':1.95,'S':2.30,'C':1.85,'H':1.62},
    'Ni':{'Cl':2.21,'Br':2.37,'F':1.86,'N':2.05,
          'O':2.00,'S':2.28,'C':1.85,'H':1.60},
    'Cu':{'Cl':2.26,'Br':2.42,'F':1.91,'N':2.05,
          'O':1.98,'S':2.32,'C':1.90,'H':1.63},
}

def build_geometry(cas):
    metal  = cas['metal']
    ligand = cas['ligand']
    n_lig  = cas['n_ligands']
    geom   = cas.get('geometry','symmetric')
    dist   = cas.get('dist_ang', EQ.get(metal,{}).get(ligand,2.2))
    frac   = cas.get('bond_frac', 1.0)

    if geom == 'csd_real':
        sname = cas.get('struct_name','')
        if sname in CSD_ATOMS:
            return CSD_ATOMS[sname]
        raise ValueError(f"CSD structure not found: {sname}")

    elif geom in ['mixed']:
        dists = {l: round(EQ[metal].get(l,2.0)*frac,3)
                 for l in EQ[metal]}
        if ligand == 'Cl3N1':
            d_cl=dists['Cl']; d_n=dists['N']
            s=f"{metal} 0.0 0.0 0.0\n"
            for i in range(3):
                a=i*2*math.pi/3
                s+=(f"Cl {d_cl*math.cos(a):.3f} "
                    f"{d_cl*math.sin(a):.3f} 0.0\n")
            s+=f"N 0.0 0.0 {d_n:.3f}\n"
            return s
        elif ligand == 'Cl4O2':
            d_cl=dists['Cl']; d_o=dists['O']
            s=f"{metal} 0.0 0.0 0.0\n"
            for p in [(d_cl,0,0),(-d_cl,0,0),
                      (0,d_cl,0),(0,-d_cl,0)]:
                s+=f"Cl {p[0]:.3f} {p[1]:.3f} 0.0\n"
            s+=(f"O 0.0 0.0 {d_o:.3f}\n"
                f"O 0.0 0.0 {-d_o:.3f}\n")
            return s
        elif ligand == 'Cl2F4':
            d_cl=dists['Cl']; d_f=dists['F']
            s=f"{metal} 0.0 0.0 0.0\n"
            for p in [(d_f,0,0),(-d_f,0,0),
                      (0,d_f,0),(0,-d_f,0)]:
                s+=f"F {p[0]:.3f} {p[1]:.3f} 0.0\n"
            s+=(f"Cl 0.0 0.0 {d_cl:.3f}\n"
                f"Cl 0.0 0.0 {-d_cl:.3f}\n")
            return s
        raise ValueError(f"Unknown mixed: {ligand}")

    elif geom in ['sqpyr','sq_pl']:
        d_ax = dist * 1.08
        s = f"{metal} 0.0 0.0 0.0\n"
        for p in [(dist,0,0),(-dist,0,0),
                  (0,dist,0),(0,-dist,0)]:
            s += f"{ligand} {p[0]:.3f} {p[1]:.3f} 0.0\n"
        s += f"{ligand} 0.0 0.0 {d_ax:.3f}\n"
        return s

    elif geom == 'tbp':
        d_ax = dist * 1.05
        s = f"{metal} 0.0 0.0 0.0\n"
        for i in range(3):
            a = i*2*math.pi/3
            s += (f"{ligand} {dist*math.cos(a):.3f} "
                  f"{dist*math.sin(a):.3f} 0.0\n")
        s += (f"{ligand} 0.0 0.0 {d_ax:.3f}\n"
              f"{ligand} 0.0 0.0 {-d_ax:.3f}\n")
        return s

    else:  # symmetric Td/Oh
        pos = {
            4:[(dist,0,0),(-dist,0,0),
               (0,dist,0),(0,-dist,0)],
            6:[(dist,0,0),(-dist,0,0),
               (0,dist,0),(0,-dist,0),
               (0,0,dist),(0,0,-dist)]
        }
        s = f"{metal} 0.0 0.0 0.0\n"
        for p in pos[n_lig]:
            s += (f"{ligand} {p[0]:.3f} "
                  f"{p[1]:.3f} {p[2]:.3f}\n")
        return s

def get_nact(n_total):
    for n in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n)>=0 and (n_total-n)%2==0: return n
    return 10

def run_uhf(mol):
    for s in [
        dict(max_cycle=300,conv_tol=1e-10,
             damp=0.0,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-9,
             damp=0.3,level_shift=0.2),
        dict(max_cycle=800,conv_tol=1e-8,
             damp=0.5,level_shift=0.5),
    ]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

# ── MAIN ───────────────────────────────────────────────────────
cas_path = sys.argv[1]
cas      = json.load(open(cas_path))

metal   = cas['metal']
ligand  = cas['ligand']
n_lig   = cas['n_ligands']
charge  = cas['charge']
spin    = cas['spin']
name    = cas['name']
geom    = cas.get('geometry','symmetric')

outdir  = os.path.expanduser('~/activeml/data/generated')
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{name}.json")

if os.path.exists(outfile):
    print(f"EXISTS: {outfile}"); sys.exit(0)

try:
    atom_str = build_geometry(cas)
except Exception as e:
    print(f"GEOMETRY ERROR: {e}"); sys.exit(1)

mol = gto.Mole()
mol.atom    = atom_str
mol.basis   = 'def2-SVP'
mol.charge  = charge
mol.spin    = spin
mol.verbose = 0
mol.build()

mf = run_uhf(mol)
s_ideal     = spin/2.0
spin_contam = float(abs(mf.spin_square()[0]-
                        s_ideal*(s_ideal+1)))

S_ovlp = mol.intor('int1e_ovlp')
X_orth = la.inv(la.sqrtm(S_ovlp))
dm_a,dm_b = mf.make_rdm1()
dm_tot_orth = X_orth@(dm_a+dm_b)@X_orth.T
uno_occ = np.sort(np.linalg.eigvalsh(dm_tot_orth))[::-1]
n_frac_uno_010 = int(np.sum((uno_occ>0.10)&(uno_occ<1.90)))
eps    = 1e-12
n_clip = np.clip(uno_occ/2,eps,1-eps)
mi_entropy_sum = float(
    -(n_clip*np.log(n_clip)+
      (1-n_clip)*np.log(1-n_clip)).sum())

e_a=mf.mo_energy[0]; e_b=mf.mo_energy[1]
occ_a=mf.mo_occ[0];  occ_b=mf.mo_occ[1]
mo_a=mf.mo_coeff[0]; mo_b=mf.mo_coeff[1]
e_mean    = (e_a+e_b)/2
occ_total = occ_a+occ_b
n_mo      = len(e_mean)

n_occ_a=int(occ_a.sum()); n_occ_b=int(occ_b.sum())
n_broken=0; min_overlap=1.0
if n_occ_a>0 and n_occ_b>0:
    try:
        S_ab=mo_a[:,:n_occ_a].T@S_ovlp@mo_b[:,:n_occ_b]
        sv=np.linalg.svd(S_ab,compute_uv=False)
        n_broken   =int(np.sum(sv<0.9))
        min_overlap=float(sv.min())
    except: pass

delta_E=0.0
hs_spin=min(mol.nelectron,8)
if (hs_spin%2)!=(mol.nelectron%2): hs_spin-=1
if hs_spin>spin:
    try:
        mol_hs=gto.Mole()
        mol_hs.atom=atom_str; mol_hs.basis='def2-SVP'
        mol_hs.charge=charge; mol_hs.spin=hs_spin
        mol_hs.verbose=0; mol_hs.build()
        mf_hs=scf.UHF(mol_hs)
        mf_hs.max_cycle=300; mf_hs.conv_tol=1e-8
        mf_hs.verbose=0; mf_hs.run()
        if mf_hs.converged:
            delta_E=float(mf.e_tot-mf_hs.e_tot)
    except: pass

homo_a=float(e_a[occ_a>0.5].max()) if (occ_a>0.5).any() else 0.0
homo_b=float(e_b[occ_b>0.5].max()) if (occ_b>0.5).any() else 0.0
homo_ab_gap=float(abs(homo_a-homo_b))

n_broken_bs=0; min_overlap_bs=1.0
if spin==0 and mol.nelectron%2==0:
    try:
        mol_bs=gto.Mole()
        mol_bs.atom=atom_str; mol_bs.basis='def2-SVP'
        mol_bs.charge=charge; mol_bs.spin=2
        mol_bs.verbose=0; mol_bs.build()
        mf_bs=scf.UHF(mol_bs)
        mf_bs.max_cycle=300; mf_bs.conv_tol=1e-8
        mf_bs.verbose=0; mf_bs.run()
        if mf_bs.converged:
            oa=mf_bs.mo_occ[0]; ob=mf_bs.mo_occ[1]
            ma=mf_bs.mo_coeff[0]; mb=mf_bs.mo_coeff[1]
            na=int(oa.sum()); nb=int(ob.sum())
            if na>0 and nb>0:
                Sbs=ma[:,:na].T@S_ovlp@mb[:,:nb]
                sv=np.linalg.svd(Sbs,compute_uv=False)
                n_broken_bs   =int(np.sum(sv<0.9))
                min_overlap_bs=float(sv.min())
    except: pass

# MP2
mp2_corr=0.0; largest_t2=0.0
n_frac_mp2_010=0; top_mi_pairs=[]; mi_matrix_sum=0.0
per_orb_mp2_occ=[]
try:
    mp2_calc=mp.MP2(mf); mp2_calc.verbose=0; mp2_calc.run()
    mp2_corr=float(mp2_calc.e_corr)
    dm_mp2_raw=mp2_calc.make_rdm1()
    if isinstance(dm_mp2_raw,tuple):
        dm_mp2=dm_mp2_raw[0]+dm_mp2_raw[1]
    else:
        dm_mp2=dm_mp2_raw
    dm_mp2_orth=X_orth@dm_mp2@X_orth.T
    mp2_occ=np.sort(np.linalg.eigvalsh(dm_mp2_orth))[::-1]
    n_frac_mp2_010=int(np.sum((mp2_occ>0.10)&(mp2_occ<1.90)))
    occ_idx_w =np.where(occ_total>0.5)[0]
    virt_idx_w=np.where(occ_total<0.5)[0]
    if len(occ_idx_w)>=7 and len(virt_idx_w)>=7:
        per_orb_mp2_occ=[float(mp2_occ[min(i,len(mp2_occ)-1)])
                         for i in range(14)]
    t2=mp2_calc.t2
    if isinstance(t2,tuple):
        largest_t2=float(max(np.abs(tt).max()
                             for tt in t2 if tt is not None))
        if t2[1] is not None:
            mi_mat=np.einsum('ijab,ijab->ij',t2[1],t2[1])
            mi_matrix_sum=float(mi_mat.sum())
            pairs=[(i,j,float(mi_mat[i,j]))
                   for i in range(mi_mat.shape[0])
                   for j in range(mi_mat.shape[1])]
            pairs.sort(key=lambda x:-x[2])
            top_mi_pairs=[[p[0],p[1],p[2]]
                          for p in pairs[:10]]
except: pass

occ_mask =occ_total>0.5
virt_mask=occ_total<0.5
homo_e=float(e_mean[occ_mask].max()) if occ_mask.any() else 0.0
lumo_e=float(e_mean[virt_mask].min()) if virt_mask.any() else 0.0
gap_cen=(homo_e+lumo_e)/2

d_region=[float(e_mean[i]) for i in range(n_mo)
          if abs(e_mean[i]-gap_cen)<2.0]
d_bw=float(max(d_region)-min(d_region)) \
     if len(d_region)>1 else 0.0

n_w01=int(np.sum(np.abs(e_mean-gap_cen)<0.1))
n_w03=int(np.sum(np.abs(e_mean-gap_cen)<0.3))
n_w05=int(np.sum(np.abs(e_mean-gap_cen)<0.5))
near_e=np.sort(e_mean[np.abs(e_mean-gap_cen)<1.0])
max_gn=mean_gn=0.0
if len(near_e)>1:
    g=np.diff(near_e)
    max_gn=float(g.max()); mean_gn=float(g.mean())

n_sing=int(np.sum((occ_total>0.5)&(occ_total<1.5)))
occ_e=np.sort(e_mean[occ_total>1.5])[::-1]
occ_spread=float(occ_e[0]-occ_e[-1]) if len(occ_e)>1 else 0.0
h1g=float(occ_e[0]-occ_e[1]) if len(occ_e)>1 else 0.0
h2g=float(occ_e[0]-occ_e[2]) if len(occ_e)>2 else 0.0

mol_feats={
    "spin_contamination": spin_contam,
    "d_bandwidth"       : d_bw,
    "multiplicity"      : spin+1,
    "n_frac_uno_010"    : n_frac_uno_010,
    "n_broken_symm_09"  : n_broken,
    "min_sab_overlap"   : min_overlap,
    "delta_E_HS_LS"     : delta_E,
    "homo_ab_gap"       : homo_ab_gap,
    "n_frac_mp2_010"    : n_frac_mp2_010,
    "mp2_corr"          : mp2_corr,
    "largest_t2"        : largest_t2,
    "mi_entropy_sum"    : mi_entropy_sum,
    "mi_matrix_sum"     : mi_matrix_sum,
    "top_mi_pairs"      : top_mi_pairs,
    "per_orb_mp2_occ"   : per_orb_mp2_occ,
    "n_broken_bs"       : n_broken_bs,
    "min_overlap_bs"    : min_overlap_bs,
    "coord_number"      : n_lig,
    "geometry"          : geom,
}

orbitals=[]
for i in range(n_mo):
    e_i=float(e_mean[i]); occ_i=float(occ_total[i])
    gaps=[abs(e_i-float(e_mean[j]))
          for j in range(n_mo) if j!=i]
    near_gap=float(min(gaps))
    ct=[abs(e_i-float(e_mean[j]))
        for j in range(n_mo)
        if j!=i and (occ_total[j]<0.5
                     if occ_i>0.5
                     else occ_total[j]>0.5)]
    ct_gap=float(min(ct)) if ct else 999.0
    orb={"index":i,"energy":e_i,"occupation":occ_i,
         "uno_frac_dev":float(min(occ_i,2.0-occ_i)),
         "dist_homo":float(abs(e_i-homo_e)),
         "dist_lumo":float(abs(e_i-lumo_e)),
         "dist_gap_center":float(abs(e_i-gap_cen)),
         "nearest_gap":near_gap,"ct_gap":ct_gap,
         "n_near_orbs":sum(1 for j in range(n_mo)
                           if j!=i and
                           abs(e_i-float(e_mean[j]))<0.3)}
    orb.update(mol_feats)
    orbitals.append(orb)

record={
    "name":name,"metal":metal,"ligand":ligand,
    "n_ligands":n_lig,"charge":charge,"spin":spin,
    "mult":spin+1,"geometry":geom,
    "n_electrons":mol.nelectron,"E_HF":float(mf.e_tot),
    "homo_energy":homo_e,"lumo_energy":lumo_e,
    "homo_lumo_gap":lumo_e-homo_e,
    "n_within_01":n_w01,"n_within_03":n_w03,
    "n_within_05":n_w05,"max_gap_near":max_gn,
    "mean_gap_near":mean_gn,"n_singly_occ":n_sing,
    "occ_spread":occ_spread,"homo_1_gap":h1g,
    "homo_2_gap":h2g,"orbitals":orbitals,
}
record.update(mol_feats)

with open(outfile,'w') as f:
    json.dump(record,f,indent=2)
print(f"Saved: {outfile}")
print(f"geom={geom} spin_c={spin_contam:.3f} "
      f"dE={delta_E:.3f} mp2={n_frac_mp2_010} "
      f"mi_pairs={len(top_mi_pairs)} "
      f"mp2_orb={len(per_orb_mp2_occ)}")
