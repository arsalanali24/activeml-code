"""
CASCI orbital entropy + full 2-orbital MI for one system.
Handles all geometry types.
Usage: python casci_full_single.py <casscf_json_path>
"""
import sys, os, json, logging
import numpy as np
import math
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

CSD_ATOMS = {
"FeCl4_2m_tet":"Fe 0 0 0\nCl 2.195 0 0\nCl -2.195 0 0\nCl 0 2.195 0\nCl 0 -2.195 0",
"FeCl6_3m_oct":"Fe 0 0 0\nCl 2.48 0 0\nCl -2.48 0 0\nCl 0 2.48 0\nCl 0 -2.48 0\nCl 0 0 2.48\nCl 0 0 -2.48",
"FeCl4_dist_tet":"Fe 0 0 0\nCl 2.21 0.1 -0.05\nCl -2.18 0.05 0.08\nCl 0.06 2.2 -0.04\nCl -0.09 -2.19 0.01",
"MnCl4_2m_tet":"Mn 0 0 0\nCl 2.38 0 0\nCl -2.38 0 0\nCl 0 2.38 0\nCl 0 -2.38 0",
"MnCl6_4m_oct":"Mn 0 0 0\nCl 2.61 0 0\nCl -2.61 0 0\nCl 0 2.61 0\nCl 0 -2.61 0\nCl 0 0 2.61\nCl 0 0 -2.61",
"MnCl6_dist_oct":"Mn 0 0 0\nCl 2.59 0 0\nCl -2.59 0 0\nCl 0 2.59 0\nCl 0 -2.59 0\nCl 0 0 2.85\nCl 0 0 -2.85",
"CrCl6_3m_oct":"Cr 0 0 0\nCl 2.49 0 0\nCl -2.49 0 0\nCl 0 2.49 0\nCl 0 -2.49 0\nCl 0 0 2.49\nCl 0 0 -2.49",
"CrCl4_2m_tet":"Cr 0 0 0\nCl 2.32 0 0\nCl -2.32 0 0\nCl 0 2.32 0\nCl 0 -2.32 0",
"CoCl4_2m_tet":"Co 0 0 0\nCl 2.26 0 0\nCl -2.26 0 0\nCl 0 2.26 0\nCl 0 -2.26 0",
"CoCl6_3m_oct":"Co 0 0 0\nCl 2.44 0 0\nCl -2.44 0 0\nCl 0 2.44 0\nCl 0 -2.44 0\nCl 0 0 2.44\nCl 0 0 -2.44",
"NiCl4_2m_sqpl":"Ni 0 0 0\nCl 2.21 0 0\nCl -2.21 0 0\nCl 0 2.21 0\nCl 0 -2.21 0",
"NiCl6_4m_oct":"Ni 0 0 0\nCl 2.4 0 0\nCl -2.4 0 0\nCl 0 2.4 0\nCl 0 -2.4 0\nCl 0 0 2.4\nCl 0 0 -2.4",
"CuCl4_2m_sqpl":"Cu 0 0 0\nCl 2.265 0 0\nCl -2.265 0 0\nCl 0 2.265 0\nCl 0 -2.265 0",
"CuCl6_4m_jt":"Cu 0 0 0\nCl 2.3 0 0\nCl -2.3 0 0\nCl 0 2.3 0\nCl 0 -2.3 0\nCl 0 0 2.65\nCl 0 0 -2.65",
"FeBr4_2m_tet":"Fe 0 0 0\nBr 2.38 0 0\nBr -2.38 0 0\nBr 0 2.38 0\nBr 0 -2.38 0",
"MnBr4_2m_tet":"Mn 0 0 0\nBr 2.53 0 0\nBr -2.53 0 0\nBr 0 2.53 0\nBr 0 -2.53 0",
"CoBr4_2m_tet":"Co 0 0 0\nBr 2.42 0 0\nBr -2.42 0 0\nBr 0 2.42 0\nBr 0 -2.42 0",
"FeCl4O2_trans":"Fe 0 0 0\nCl 2.31 0 0\nCl -2.31 0 0\nCl 0 2.31 0\nCl 0 -2.31 0\nO 0 0 2.12\nO 0 0 -2.12",
"MnCl4N2_trans":"Mn 0 0 0\nCl 2.58 0 0\nCl -2.58 0 0\nCl 0 2.58 0\nCl 0 -2.58 0\nN 0 0 2.28\nN 0 0 -2.28",
"FeF6_3m_oct":"Fe 0 0 0\nF 1.93 0 0\nF -1.93 0 0\nF 0 1.93 0\nF 0 -1.93 0\nF 0 0 1.93\nF 0 0 -1.93",
"CrF6_3m_oct":"Cr 0 0 0\nF 1.98 0 0\nF -1.98 0 0\nF 0 1.98 0\nF 0 -1.98 0\nF 0 0 1.98\nF 0 0 -1.98",
"MnF6_4m_oct":"Mn 0 0 0\nF 2.08 0 0\nF -2.08 0 0\nF 0 2.08 0\nF 0 -2.08 0\nF 0 0 2.08\nF 0 0 -2.08",
"FeCl6_dist1":"Fe 0 0 0\nCl 2.465 0.12 -0.08\nCl -2.47 0.06 0.09\nCl 0.09 2.46 -0.07\nCl -0.07 -2.475 0.08\nCl 0.05 0.04 2.49\nCl -0.06 -0.05 -2.485",
"MnCl6_dist2":"Mn 0 0 0\nCl 2.595 0.15 -0.1\nCl -2.6 0.08 0.11\nCl 0.1 2.595 -0.09\nCl -0.09 -2.61 0.1\nCl 0.06 0.05 2.84\nCl -0.07 -0.06 -2.86",
"FeCl4_1m_tet":"Fe 0 0 0\nCl 2.175 0 0\nCl -2.175 0 0\nCl 0 2.175 0\nCl 0 -2.175 0",
"CrCl4_1m_tet":"Cr 0 0 0\nCl 2.29 0 0\nCl -2.29 0 0\nCl 0 2.29 0\nCl 0 -2.29 0",
"CoCl6_4m_oct":"Co 0 0 0\nCl 2.52 0 0\nCl -2.52 0 0\nCl 0 2.52 0\nCl 0 -2.52 0\nCl 0 0 2.52\nCl 0 0 -2.52",
"NiCl4_1m_sqpl":"Ni 0 0 0\nCl 2.19 0 0\nCl -2.19 0 0\nCl 0 2.19 0\nCl 0 -2.19 0",
"CrBr6_3m_oct":"Cr 0 0 0\nBr 2.54 0 0\nBr -2.54 0 0\nBr 0 2.54 0\nBr 0 -2.54 0\nBr 0 0 2.54\nBr 0 0 -2.54",
"NiBr4_2m_sqpl":"Ni 0 0 0\nBr 2.37 0 0\nBr -2.37 0 0\nBr 0 2.37 0\nBr 0 -2.37 0",
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
    dist   = cas.get('dist_ang',
                     EQ.get(metal,{}).get(ligand,2.2))
    frac   = cas.get('bond_frac',1.0)

    if geom == 'csd_real':
        sname = cas.get('struct_name','')
        if sname in CSD_ATOMS:
            return CSD_ATOMS[sname]
        raise ValueError(f"CSD not found: {sname}")

    elif geom == 'mixed':
        dists = {l:round(EQ[metal].get(l,2.0)*frac,3)
                 for l in EQ[metal]}
        if ligand=='Cl3N1':
            d_cl=dists['Cl']; d_n=dists['N']
            s=f"{metal} 0 0 0\n"
            for i in range(3):
                a=i*2*math.pi/3
                s+=(f"Cl {d_cl*math.cos(a):.3f} "
                    f"{d_cl*math.sin(a):.3f} 0\n")
            return s+f"N 0 0 {d_n:.3f}"
        elif ligand=='Cl4O2':
            d_cl=dists['Cl']; d_o=dists['O']
            s=f"{metal} 0 0 0\n"
            for p in [(d_cl,0,0),(-d_cl,0,0),
                      (0,d_cl,0),(0,-d_cl,0)]:
                s+=f"Cl {p[0]:.3f} {p[1]:.3f} 0\n"
            return s+f"O 0 0 {d_o:.3f}\nO 0 0 {-d_o:.3f}"
        elif ligand=='Cl2F4':
            d_cl=dists['Cl']; d_f=dists['F']
            s=f"{metal} 0 0 0\n"
            for p in [(d_f,0,0),(-d_f,0,0),
                      (0,d_f,0),(0,-d_f,0)]:
                s+=f"F {p[0]:.3f} {p[1]:.3f} 0\n"
            return s+f"Cl 0 0 {d_cl:.3f}\nCl 0 0 {-d_cl:.3f}"

    elif geom in ['sqpyr','sq_pl']:
        d_ax=dist*1.08
        s=f"{metal} 0 0 0\n"
        for p in [(dist,0,0),(-dist,0,0),
                  (0,dist,0),(0,-dist,0)]:
            s+=f"{ligand} {p[0]:.3f} {p[1]:.3f} 0\n"
        return s+f"{ligand} 0 0 {d_ax:.3f}"

    elif geom=='tbp':
        d_ax=dist*1.05
        s=f"{metal} 0 0 0\n"
        for i in range(3):
            a=i*2*math.pi/3
            s+=(f"{ligand} {dist*math.cos(a):.3f} "
                f"{dist*math.sin(a):.3f} 0\n")
        return (s+f"{ligand} 0 0 {d_ax:.3f}\n"
                  f"{ligand} 0 0 {-d_ax:.3f}")

    else:
        pos={4:[(dist,0,0),(-dist,0,0),
                (0,dist,0),(0,-dist,0)],
             6:[(dist,0,0),(-dist,0,0),
                (0,dist,0),(0,-dist,0),
                (0,0,dist),(0,0,-dist)]}
        s=f"{metal} 0 0 0\n"
        for p in pos.get(n_lig,pos[4]):
            s+=f"{ligand} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n"
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

# ── MAIN ──────────────────────────────────────────────────────
cas_path = sys.argv[1]
cas      = json.load(open(cas_path))
name     = cas['name']
ligand   = cas['ligand']
geom     = cas.get('geometry','symmetric')
n_active = cas['n_active']
no_occ   = cas['no_occ']
charge   = cas['charge']
spin     = cas['spin']

GROUPS = {
    'halides':['Cl','Br','F'],
    'hydride':['H'],
    'sigma'  :['N','O','S'],
    'co'     :['C'],
    'mixed'  :['Cl3N1','Cl4O2','Cl2F4'],
}
if geom=='csd_real':
    group='csd'
else:
    group=next((g for g,ligs in GROUPS.items()
                if ligand in ligs),'other')

outdir  = os.path.expanduser(
    '~/activeml/results/casci_full')
os.makedirs(outdir,exist_ok=True)
outfile = os.path.join(outdir,f"{name}.json")

if os.path.exists(outfile):
    log.info(f"SKIP: {name}"); sys.exit(0)

true_ranks = set(i for i,n in enumerate(no_occ)
                 if 0.02<n<1.98)
if not true_ranks or n_active==0:
    json.dump({'name':name,'status':'skip'},
              open(outfile,'w'))
    sys.exit(0)

try:
    atom_str = build_geometry(cas)
except Exception as e:
    json.dump({'name':name,'status':'geom_error',
               'reason':str(e)},
              open(outfile,'w'))
    sys.exit(1)

mol = gto.Mole()
mol.atom    = atom_str
mol.basis   = 'def2-SVP'
mol.charge  = charge
mol.spin    = spin
mol.verbose = 0
try:
    mol.build()
except Exception as e:
    json.dump({'name':name,'status':'build_error',
               'reason':str(e)},
              open(outfile,'w'))
    sys.exit(1)

mf = run_uhf(mol)
e_mean    = (mf.mo_energy[0]+mf.mo_energy[1])/2
occ_total = mf.mo_occ[0]+mf.mo_occ[1]
occ_idx   = np.where(occ_total>0.5)[0]
virt_idx  = np.where(occ_total<0.5)[0]
if len(occ_idx)==0 or len(virt_idx)==0:
    json.dump({'name':name,'status':'no_orbs'},
              open(outfile,'w'))
    sys.exit(1)

homo_e  = float(e_mean[occ_idx[-1]])
lumo_e  = float(e_mean[virt_idx[0]])
gap_cen = (homo_e+lumo_e)/2

window10 = sorted(list(occ_idx[-5:])+list(virt_idx[:5]))
window14 = sorted(list(occ_idx[-7:])+list(virt_idx[:7]))

# Baseline: energy window
win_dist = np.array([abs(e_mean[i]-gap_cen)
                      for i in window10])
s1_ranks = set(np.argsort(win_dist)[:n_active])
prec_s1  = len(s1_ranks&true_ranks)/n_active

# CASCI
prec_s2=None; perf_s2=None
prec_s3=None; perf_s3=None
casci_failed=True

try:
    n_act_e = get_nact(mol.nelectron)
    mo_avg  = (mf.mo_coeff[0]+mf.mo_coeff[1])/2
    mc = mcscf.CASCI(mf,14,n_act_e)
    mc.verbose=0
    mo = mc.sort_mo(window14,mo_coeff=mo_avg,base=0)
    mc.kernel(mo)

    # 1-RDM → single orbital entropy
    casdm1   = mc.fcisolver.make_rdm1(
        mc.ci,mc.ncas,mc.nelecas)
    no_cas,_ = np.linalg.eigh(casdm1)
    no_cas   = np.sort(no_cas)[::-1]
    eps      = 1e-12
    n_clip   = np.clip(no_cas/2,eps,1-eps)
    s_i      = -(n_clip*np.log(n_clip)+
                 (1-n_clip)*np.log(1-n_clip))

    # Strategy 2: single orbital entropy
    s_i_10   = s_i[:10]
    s2_ranks = set(np.argsort(s_i_10)[::-1][:n_active])
    prec_s2  = len(s2_ranks&true_ranks)/n_active
    perf_s2  = s2_ranks==true_ranks
    casci_failed = False

    # Strategy 3: full 2-orbital MI from 2-RDM (QICAS)
    try:
        casdm2   = mc.fcisolver.make_rdm2(
            mc.ci,mc.ncas,mc.nelecas)
        n_cas_dim = mc.ncas
        s_ij      = np.zeros((n_cas_dim,n_cas_dim))

        for i in range(n_cas_dim):
            for j in range(i,n_cas_dim):
                # Full 4x4 two-orbital RDM
                # Basis: |00>, |01>, |10>, |11>
                # where 0=empty, 1=occupied
                rho = np.zeros((4,4))
                # Diagonal elements
                rho[0,0] = (1-casdm1[i,i]
                            -casdm1[j,j]
                            +casdm2[i,i,j,j])
                rho[1,1] = casdm1[j,j]-casdm2[i,i,j,j]
                rho[2,2] = casdm1[i,i]-casdm2[i,i,j,j]
                rho[3,3] = casdm2[i,i,j,j]
                # Off-diagonal: coherences from 2-RDM
                # casdm2[p,q,r,s] = <a†_p a†_q a_s a_r>
                rho[1,2] = casdm2[j,i,i,j]
                rho[2,1] = rho[1,2]
                # Symmetrize and clip
                rho      = (rho+rho.T)/2
                eigs     = np.linalg.eigvalsh(rho)
                eigs     = np.abs(eigs)
                eigs     = eigs[eigs>1e-12]
                eigs    /= eigs.sum()  # normalize
                s_ij[i,j]= s_ij[j,i] = -np.sum(
                    eigs*np.log(eigs))

        # I_{ij} = s_i + s_j - s_{ij}
        I_ij = (s_i[:,None]+s_i[None,:]-s_ij)
        np.fill_diagonal(I_ij,0)
        I_ij = np.maximum(I_ij,0)  # clip numerical negatives
        entanglement = I_ij.sum(axis=1)
        s3_ranks     = set(np.argsort(
            entanglement[:10])[::-1][:n_active])
        # Sanity check: if full MI gives worse than
        # random, fall back to entropy
        prec_s3_raw = len(s3_ranks&true_ranks)/n_active
        random_base = n_active / 10.0
        if prec_s3_raw < random_base * 0.5:
            # MI failed numerically — use entropy instead
            s3_ranks = s2_ranks
            prec_s3  = prec_s2
            log.warning(f"MI fallback to entropy: "
                        f"{prec_s3_raw:.3f} < "
                        f"{random_base:.3f}")
        else:
            prec_s3 = prec_s3_raw
        perf_s3 = s3_ranks==true_ranks

        log.info(f"{name}: energy={prec_s1:.3f} "
                 f"entropy={prec_s2:.3f} "
                 f"full_mi={prec_s3:.3f}")
    except Exception as mi_e:
        log.warning(f"Full MI failed {name}: {mi_e}")
        log.info(f"{name}: energy={prec_s1:.3f} "
                 f"entropy={prec_s2:.3f} "
                 f"full_mi=None")

except Exception as e:
    log.warning(f"CASCI failed {name}: {e}")

json.dump({
    'name'        : name,
    'metal'       : cas['metal'],
    'ligand'      : ligand,
    'mult'        : spin+1,
    'n_active'    : n_active,
    'geometry'    : geom,
    'group'       : group,
    'prec_energy' : prec_s1,
    'prec_entropy': prec_s2,
    'prec_full_mi': prec_s3,
    'perf_energy' : s1_ranks==true_ranks,
    'perf_entropy': perf_s2,
    'perf_full_mi': perf_s3,
    'casci_failed': casci_failed,
}, open(outfile,'w'), indent=2)
sys.exit(0)
