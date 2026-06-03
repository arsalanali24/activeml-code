"""Run CASCI entropy for one system by index."""
import sys, os, json, glob, logging
import numpy as np
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

def build_geometry(metal, ligand, n_lig, dist):
    pos = {
        4:[(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6:[(dist,0,0),(-dist,0,0),(0,dist,0),
           (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    s = f"{metal}  0.000  0.000  0.000\n"
    for p in pos[n_lig]:
        s += f"{ligand}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}\n"
    return s

def get_active_electrons(n_total):
    for n in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n)>=0 and (n_total-n)%2==0:
            return n
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
        mf = scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

# ── BUILD JOB LIST ────────────────────────────────────────────
casscf_dir = os.path.expanduser('~/activeml/data/generated300')
target_ligs = ['Cl','Br','F']
target_n    = 30

counts = {l:0 for l in target_ligs}
ALL_JOBS = []

for cas_path in sorted(glob.glob(f'{casscf_dir}/*.json')):
    cas = json.load(open(cas_path))
    if cas.get('status')!='ok': continue
    if cas.get('corr_energy',0)>=0: continue
    lig = cas.get('ligand','Cl')
    if lig not in target_ligs: continue
    if counts[lig] >= target_n: continue
    if cas['n_active'] == 0: continue
    ALL_JOBS.append(cas)
    counts[lig] += 1
    if all(v>=target_n for v in counts.values()):
        break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'count':
        print(len(ALL_JOBS))
        sys.exit(0)

    job_idx = int(sys.argv[1])
    if job_idx >= len(ALL_JOBS):
        sys.exit(1)

    cas      = ALL_JOBS[job_idx]
    metal    = cas['metal']
    ligand   = cas['ligand']
    n_lig    = cas['n_ligands']
    charge   = cas['charge']
    spin     = cas['spin']
    dist     = cas['dist_ang']
    n_active = cas['n_active']
    no_occ   = cas['no_occ']
    name     = f"{metal}_{ligand}{n_lig}_chg{charge}_spin{spin}"

    outdir = os.path.expanduser('~/activeml/results/casci')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        log.info(f"SKIP: {name}")
        sys.exit(0)

    log.info(f"Starting: {name} n_active={n_active}")

    mol = gto.Mole()
    mol.atom    = build_geometry(metal,ligand,n_lig,dist)
    mol.basis   = 'def2-SVP'
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = 0
    mol.build()

    mf = run_uhf(mol)
    e_mean    = (mf.mo_energy[0]+mf.mo_energy[1])/2
    occ_total = mf.mo_occ[0]+mf.mo_occ[1]
    occ_idx   = np.where(occ_total>0.5)[0]
    virt_idx  = np.where(occ_total<0.5)[0]

    # 10-window for baseline
    window10 = sorted(list(occ_idx[-5:])+list(virt_idx[:5]))
    # 14-window balanced for CASCI
    window14 = sorted(list(occ_idx[-7:])+list(virt_idx[:7]))

    # True active ranks in 10-window
    true_ranks = set(i for i,n in enumerate(no_occ)
                     if 0.02<n<1.98)

    homo_e  = float(e_mean[occ_idx[-1]])
    lumo_e  = float(e_mean[virt_idx[0]])
    gap_cen = (homo_e+lumo_e)/2

    # Baseline: energy window
    win_dist = np.array([abs(e_mean[i]-gap_cen)
                          for i in window10])
    s1_ranks = set(np.argsort(win_dist)[:n_active])
    prec_s1  = len(s1_ranks&true_ranks)/n_active \
               if n_active>0 else 1.0

    # CASCI entropy
    prec_s2 = None
    perf_s2 = None
    casci_failed = True

    try:
        n_act_e = get_active_electrons(mol.nelectron)
        mo_avg  = (mf.mo_coeff[0]+mf.mo_coeff[1])/2
        mc = mcscf.CASCI(mf, 14, n_act_e)
        mc.verbose = 0
        mo = mc.sort_mo(window14, mo_coeff=mo_avg, base=0)
        mc.kernel(mo)

        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_occ_cas,_ = np.linalg.eigh(casdm1)
        no_occ_cas   = np.sort(no_occ_cas)[::-1]
        eps    = 1e-12
        n_clip = np.clip(no_occ_cas/2, eps, 1-eps)
        s_i    = -(n_clip*np.log(n_clip)+
                    (1-n_clip)*np.log(1-n_clip))

        # Map entropy to 10-window ranks
        # window14[0:10] covers same range as window10
        # Use first 10 positions of 14-window
        s_i_10   = s_i[:10]
        s2_ranks = set(np.argsort(s_i_10)[::-1][:n_active])
        prec_s2  = len(s2_ranks&true_ranks)/n_active \
                   if n_active>0 else 1.0
        perf_s2  = s2_ranks==true_ranks
        casci_failed = False
        log.info(f"  energy={prec_s1:.3f} "
                 f"entropy={prec_s2:.3f}")

    except Exception as e:
        log.warning(f"CASCI failed: {type(e).__name__}: {e}")

    result = {
        'name'        : name,
        'metal'       : metal,
        'ligand'      : ligand,
        'mult'        : spin+1,
        'n_active'    : n_active,
        'prec_energy' : prec_s1,
        'prec_entropy': prec_s2,
        'perf_energy' : s1_ranks==true_ranks,
        'perf_entropy': perf_s2,
        'casci_failed': casci_failed,
    }

    with open(outfile,'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved: {outfile}")
