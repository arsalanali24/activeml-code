"""
CASCI orbital entropy test across ALL ligand groups.
Tests: halides, hydride, mixed, sigma-donors, CO.
Runs one system per job index.
"""
import sys, os, json, glob, logging
import numpy as np
from pyscf import gto, scf, mcscf
import math

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

casscf_dir = os.path.expanduser('~/activeml/data/generated300')

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

def build_sym(metal, ligand, n_lig, dist):
    pos={4:[(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
         6:[(dist,0,0),(-dist,0,0),(0,dist,0),
            (0,-dist,0),(0,0,dist),(0,0,-dist)]}
    s=f"{metal}  0.000  0.000  0.000\n"
    for p in pos[n_lig]:
        s+=f"{ligand}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}\n"
    return s

EQ={'Fe':{'Cl':2.18,'Br':2.35,'F':1.85,'N':2.10,
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
          'O':1.98,'S':2.32,'C':1.90,'H':1.63}}

def build_mixed(metal, mix_type, frac):
    eq=EQ[metal]
    dists={lig:round(eq.get(lig,2.0)*frac,3)
           for lig in eq}
    if mix_type=='Cl3N1':
        d_cl=dists['Cl']; d_n=dists['N']
        s=f"{metal}  0.000  0.000  0.000\n"
        for i in range(3):
            a=i*2*math.pi/3
            s+=(f"Cl  {d_cl*math.cos(a):.3f}  "
                f"{d_cl*math.sin(a):.3f}  0.000\n")
        s+=f"N  0.000  0.000  {d_n:.3f}\n"
        return s
    elif mix_type=='Cl4O2':
        d_cl=dists['Cl']; d_o=dists['O']
        s=f"{metal}  0.000  0.000  0.000\n"
        for p in [(d_cl,0,0),(-d_cl,0,0),
                  (0,d_cl,0),(0,-d_cl,0)]:
            s+=f"Cl  {p[0]:.3f}  {p[1]:.3f}  0.000\n"
        s+=(f"O  0.000  0.000  {d_o:.3f}\n"
            f"O  0.000  0.000  {-d_o:.3f}\n")
        return s
    elif mix_type=='Cl2F4':
        d_cl=dists['Cl']; d_f=dists['F']
        s=f"{metal}  0.000  0.000  0.000\n"
        for p in [(d_f,0,0),(-d_f,0,0),
                  (0,d_f,0),(0,-d_f,0)]:
            s+=f"F  {p[0]:.3f}  {p[1]:.3f}  0.000\n"
        s+=(f"Cl  0.000  0.000  {d_cl:.3f}\n"
            f"Cl  0.000  0.000  {-d_cl:.3f}\n")
        return s
    raise ValueError(f"Unknown: {mix_type}")

def build_geometry_from_cas(cas):
    metal  = cas['metal']
    ligand = cas['ligand']
    n_lig  = cas['n_ligands']
    dist   = cas['dist_ang']
    geom   = cas.get('geometry','symmetric')
    frac   = cas.get('bond_frac',1.0)
    if geom == 'mixed':
        return build_mixed(metal, ligand, frac)
    else:
        return build_sym(metal, ligand, n_lig, dist)

def test_system(cas):
    metal    = cas['metal']
    ligand   = cas['ligand']
    charge   = cas['charge']
    spin     = cas['spin']
    n_active = cas['n_active']
    no_occ   = cas['no_occ']
    name     = cas['name']

    # True active ranks within 10-window
    true_ranks = set(i for i,n in enumerate(no_occ)
                     if 0.02<n<1.98)
    if not true_ranks or n_active==0:
        return None

    mol = gto.Mole()
    mol.atom    = build_geometry_from_cas(cas)
    mol.basis   = 'def2-SVP'
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = 0
    try:
        mol.build()
    except Exception as e:
        log.warning(f"mol.build failed: {e}")
        return None

    mf = run_uhf(mol)
    e_mean    = (mf.mo_energy[0]+mf.mo_energy[1])/2
    occ_total = mf.mo_occ[0]+mf.mo_occ[1]
    occ_idx   = np.where(occ_total>0.5)[0]
    virt_idx  = np.where(occ_total<0.5)[0]
    if len(occ_idx)==0 or len(virt_idx)==0:
        return None

    homo_e  = float(e_mean[occ_idx[-1]])
    lumo_e  = float(e_mean[virt_idx[0]])
    gap_cen = (homo_e+lumo_e)/2

    # 10-orbital window (5 occ + 5 virt)
    window10 = sorted(list(occ_idx[-5:])+list(virt_idx[:5]))
    # 14-orbital window (7 occ + 7 virt)
    window14 = sorted(list(occ_idx[-7:])+list(virt_idx[:7]))

    # Baseline: energy window
    win_dist = np.array([abs(e_mean[i]-gap_cen)
                          for i in window10])
    s1_ranks = set(np.argsort(win_dist)[:n_active])
    prec_s1  = len(s1_ranks&true_ranks)/n_active

    # CASCI entropy
    prec_s2 = None; perf_s2 = None
    casci_failed = True
    try:
        n_act_e = get_nact(mol.nelectron)
        mo_avg  = (mf.mo_coeff[0]+mf.mo_coeff[1])/2
        mc = mcscf.CASCI(mf, 14, n_act_e)
        mc.verbose = 0
        mo = mc.sort_mo(window14, mo_coeff=mo_avg, base=0)
        mc.kernel(mo)

        casdm1  = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_cas,_ = np.linalg.eigh(casdm1)
        no_cas   = np.sort(no_cas)[::-1]
        eps      = 1e-12
        n_clip   = np.clip(no_cas/2, eps, 1-eps)
        s_i      = -(n_clip*np.log(n_clip)+
                     (1-n_clip)*np.log(1-n_clip))

        s_i_10   = s_i[:10]
        s2_ranks = set(np.argsort(s_i_10)[::-1][:n_active])
        prec_s2  = len(s2_ranks&true_ranks)/n_active
        perf_s2  = s2_ranks==true_ranks
        casci_failed = False
    except Exception as e:
        log.warning(f"CASCI failed: {type(e).__name__}: {e}")

    return {
        'name'        : name,
        'metal'       : metal,
        'ligand'      : ligand,
        'mult'        : spin+1,
        'n_active'    : n_active,
        'geometry'    : cas.get('geometry','symmetric'),
        'prec_energy' : prec_s1,
        'prec_entropy': prec_s2,
        'perf_energy' : s1_ranks==true_ranks,
        'perf_entropy': perf_s2,
        'casci_failed': casci_failed,
    }

# ── BUILD JOB LIST ─────────────────────────────────────────────
# 25 systems per group for balanced comparison
TARGET_N = 25
GROUPS = {
    'halides' : ['Cl','Br','F'],
    'hydride' : ['H'],
    'mixed'   : ['Cl3N1','Cl4O2','Cl2F4'],
    'sigma'   : ['N','O','S'],
    'co'      : ['C'],
}

counts  = {g:0 for g in GROUPS}
ALL_JOBS = []
seen     = set()

for cas_path in sorted(glob.glob(f'{casscf_dir}/*.json')):
    try:
        cas = json.load(open(cas_path))
        if cas.get('status')!='ok': continue
        if cas.get('corr_energy',0)>=0: continue
        if cas.get('n_active',0)==0: continue
        lig = cas.get('ligand','')
        grp = next((g for g,ligs in GROUPS.items()
                    if lig in ligs), None)
        if grp is None: continue
        if counts[grp] >= TARGET_N: continue
        name = cas['name']
        if name in seen: continue
        seen.add(name)
        ALL_JOBS.append(cas_path)
        counts[grp] += 1
        if all(v>=TARGET_N for v in counts.values()):
            break
    except:
        continue

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_grp = defaultdict(int)
        for path in ALL_JOBS:
            r=json.load(open(path))
            lig=r.get('ligand','')
            grp=next((g for g,ligs in GROUPS.items()
                      if lig in ligs),'?')
            by_grp[grp]+=1
        print(f"Total jobs: {len(ALL_JOBS)}")
        for g,n in sorted(by_grp.items()):
            print(f"  {g}: {n}")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>=len(ALL_JOBS): sys.exit(1)

    cas_path = ALL_JOBS[idx]
    cas      = json.load(open(cas_path))
    name     = cas['name']
    lig      = cas.get('ligand','')
    grp      = next((g for g,ligs in GROUPS.items()
                     if lig in ligs),'?')

    outdir = os.path.expanduser(
        '~/activeml/results/casci_all')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        log.info(f"SKIP: {name}"); sys.exit(0)

    log.info(f"[{idx}] {name} grp={grp} "
             f"n_active={cas['n_active']}")

    result = test_system(cas)
    if result is None:
        json.dump({'name':name,'status':'failed'},
                  open(outfile,'w'))
        sys.exit(1)

    result['group'] = grp
    json.dump(result, open(outfile,'w'), indent=2)
    log.info(f"  energy={result['prec_energy']:.3f} "
             f"entropy={result.get('prec_entropy','N/A')}")
    sys.exit(0)
