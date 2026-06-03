"""
CASCI orbital entropy for orbital identification.
Fixed: balanced 7occ+7virt window, sort_mo for CASCI.
"""
import sys, os, json, logging
import numpy as np
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

LIG_CHARGE = {'Cl':-1,'Br':-1,'F':-1,'I':-1,
              'N':-3,'O':-2,'S':-2,'C':-4,'H':-1}

def build_geometry(metal, ligand, n_lig, dist):
    pos = {
        4:[(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6:[(dist,0,0),(-dist,0,0),(0,dist,0),
           (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    s = f"{metal}  0.000  0.000  0.000\n"
    for p in pos[n_lig]:
        s += (f"{ligand}  {p[0]:.3f}  "
              f"{p[1]:.3f}  {p[2]:.3f}\n")
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

def get_balanced_window(mf, n_occ_win=7, n_virt_win=7):
    """
    Get balanced window: n_occ_win highest occupied
    + n_virt_win lowest virtual MOs.
    Returns sorted list of orbital indices.
    """
    occ_total = mf.mo_occ[0] + mf.mo_occ[1]
    occ_idx   = np.where(occ_total > 0.5)[0]
    virt_idx  = np.where(occ_total < 0.5)[0]
    occ_win   = list(occ_idx[-n_occ_win:])
    virt_win  = list(virt_idx[:n_virt_win])
    return sorted(occ_win + virt_win)

def run_casci_get_entropy(mf, mol, window14):
    """
    Run CASCI on window14 orbitals.
    Returns (no_occ, s_i) or (None, None) on failure.
    """
    n_act_e = get_active_electrons(mol.nelectron)
    mo_avg  = (mf.mo_coeff[0]+mf.mo_coeff[1])/2

    try:
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
        return no_occ_cas, s_i

    except Exception as e:
        log.warning(f"CASCI failed: {type(e).__name__}: {e}")
        return None, None

def test_system(metal, ligand, n_lig, charge,
                spin, dist, true_n_active, true_no_occ):
    """Test orbital identification for one system."""
    mol = gto.Mole()
    mol.atom    = build_geometry(metal,ligand,n_lig,dist)
    mol.basis   = 'def2-SVP'
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = 0
    mol.build()

    mf = run_uhf(mol)
    if not mf.converged:
        log.warning("UHF not converged")

    e_mean    = (mf.mo_energy[0]+mf.mo_energy[1])/2
    occ_total = mf.mo_occ[0]+mf.mo_occ[1]

    # Get 10-orbital window for comparison
    occ_idx  = np.where(occ_total>0.5)[0]
    virt_idx = np.where(occ_total<0.5)[0]
    occ_win10  = list(occ_idx[-5:])
    virt_win10 = list(virt_idx[:5])
    window10   = sorted(occ_win10 + virt_win10)

    # True active ranks within 10-window
    true_ranks = set(i for i,n in enumerate(true_no_occ)
                     if 0.02 < n < 1.98)
    if not true_ranks:
        return None

    # ── Baseline: energy window ────────────────────────
    homo_e  = float(e_mean[occ_idx[-1]])
    lumo_e  = float(e_mean[virt_idx[0]])
    gap_cen = (homo_e+lumo_e)/2
    win_dist = np.array([abs(e_mean[i]-gap_cen)
                          for i in window10])
    s1_ranks = set(np.argsort(win_dist)[:true_n_active])
    prec_s1  = len(s1_ranks&true_ranks)/true_n_active

    # ── CASCI entropy: 14-orbital balanced window ─────
    window14 = get_balanced_window(mf, 7, 7)
    no_occ_cas, s_i = run_casci_get_entropy(
        mf, mol, window14)

    if s_i is None:
        return {'prec_energy' : prec_s1,
                'prec_entropy': None,
                'prec_noon'   : None,
                'perf_energy' : s1_ranks==true_ranks,
                'perf_entropy': None,
                'perf_noon'   : None,
                'casci_failed': True}

    # Map 14-window entropy back to 10-window ranks
    # The 10-window is the inner 10 orbitals of 14-window
    # (5 occ + 5 virt = middle portion)
    # Find which 14-window positions correspond to 10-window
    w14_set = set(window14)
    w10_set = set(window10)
    common  = sorted(w14_set & w10_set,
                     key=lambda i: e_mean[i])

    if len(common) >= true_n_active:
        # Get ranks within 14-window for common orbitals
        w14_sorted = sorted(window14,
                            key=lambda i: e_mean[i])
        w14_rank   = {orb:rank
                      for rank,orb in enumerate(w14_sorted)}
        common_ranks_in_14 = [w14_rank[o] for o in common]
        s_i_common = s_i[common_ranks_in_14]

        # Select top n_active by entropy within common
        top_common = np.argsort(s_i_common)[::-1][
                     :true_n_active]
        # Map back to 10-window ranks
        w10_sorted = sorted(window10,
                            key=lambda i: e_mean[i])
        w10_rank   = {orb:rank
                      for rank,orb in enumerate(w10_sorted)}
        s2_ranks   = set(w10_rank[common[i]]
                         for i in top_common
                         if common[i] in w10_rank)
    else:
        # Fallback: use full 14-window entropy
        s2_ranks = set(np.argsort(s_i)[::-1][:true_n_active])

    if len(s2_ranks) < true_n_active:
        # Fill with energy window if needed
        remaining = [r for r in range(10)
                     if r not in s2_ranks]
        s2_ranks |= set(remaining[:true_n_active-len(s2_ranks)])

    prec_s2 = len(s2_ranks&true_ranks)/true_n_active

    # NOON-based selection
    if no_occ_cas is not None:
        noon_frac = np.minimum(no_occ_cas, 2.0-no_occ_cas)
        s3_ranks  = set(np.argsort(
            noon_frac)[::-1][:true_n_active])
        prec_s3   = len(s3_ranks&true_ranks)/true_n_active
    else:
        prec_s3 = None

    return {
        'prec_energy' : prec_s1,
        'prec_entropy': prec_s2,
        'prec_noon'   : prec_s3,
        'perf_energy' : s1_ranks==true_ranks,
        'perf_entropy': s2_ranks==true_ranks,
        'casci_failed': False,
    }


if __name__ == "__main__":
    casscf_dir = os.path.expanduser(
        '~/activeml/data/generated300')
    target_n   = int(sys.argv[1]) \
                 if len(sys.argv)>1 else 30

    import glob
    target_ligs = ['Cl','Br','F']
    counts = {l:0 for l in target_ligs}
    results = []

    log.info(f"Target: {target_n} systems per ligand")

    for cas_path in sorted(glob.glob(
            f'{casscf_dir}/*.json')):
        try:
            cas = json.load(open(cas_path))
            if cas.get('status')!='ok': continue
            if cas.get('corr_energy',0)>=0: continue
            ligand = cas.get('ligand','Cl')
            if ligand not in target_ligs: continue
            if counts[ligand] >= target_n: continue

            metal    = cas['metal']
            charge   = cas['charge']
            spin     = cas['spin']
            n_lig    = cas['n_ligands']
            dist     = cas['dist_ang']
            n_active = cas['n_active']
            no_occ   = cas['no_occ']
            if n_active == 0: continue

            log.info(
                f"[{sum(counts.values())}/{target_n*3}] "
                f"{metal}_{ligand}{n_lig}_chg{charge}"
                f"_spin{spin} n_active={n_active}")

            result = test_system(
                metal,ligand,n_lig,charge,
                spin,dist,n_active,no_occ)
            if result is None: continue

            result.update({
                'metal'   : metal,
                'ligand'  : ligand,
                'mult'    : spin+1,
                'n_active': n_active,
            })
            results.append(result)
            counts[ligand] += 1

            if len(results) % 5 == 0:
                e = np.mean([r['prec_energy']
                              for r in results])
                s = [r['prec_entropy'] for r in results
                     if r['prec_entropy'] is not None]
                s_mean = np.mean(s) if s else 0
                log.info(
                    f"  avg energy={e:.3f} "
                    f"entropy={s_mean:.3f} "
                    f"n={len(results)}")

        except Exception as e:
            log.warning(f"System error: {e}")
            continue

        if all(v>=target_n for v in counts.values()):
            break

    # Results
    outdir = os.path.expanduser('~/activeml/results')
    os.makedirs(outdir, exist_ok=True)
    json.dump(results,
              open(f'{outdir}/casci_entropy.json','w'),
              indent=2)

    e_vals = [r['prec_energy'] for r in results]
    s_vals = [r['prec_entropy'] for r in results
              if r['prec_entropy'] is not None]
    n_tot  = len(results)
    n_fail = sum(1 for r in results
                 if r.get('casci_failed',False))

    print(f"\n{'='*55}")
    print(f"CASCI ENTROPY RESULTS  (n={n_tot}, failed={n_fail})")
    print(f"{'='*55}")
    print(f"{'Strategy':<30} {'Prec':>8} {'Perfect':>10}")
    print(f"{'-'*50}")
    e_perf = sum(1 for r in results if r['perf_energy'])
    s_perf = sum(1 for r in results
                 if r.get('perf_entropy'))
    print(f"{'Energy window':<30} "
          f"{np.mean(e_vals):>8.3f} "
          f"{e_perf}/{n_tot} ({e_perf/n_tot:.1%})")
    if s_vals:
        print(f"{'CASCI entropy':<30} "
              f"{np.mean(s_vals):>8.3f} "
              f"{s_perf}/{n_tot} ({s_perf/n_tot:.1%})")
        print(f"\nImprovement: "
              f"{np.mean(s_vals)-np.mean(e_vals):+.3f}")

    print(f"\nBy multiplicity:")
    for mult in [1,2,3,4,5,6]:
        g = [r for r in results if r['mult']==mult]
        if len(g)<2: continue
        e = np.mean([r['prec_energy'] for r in g])
        s = [r['prec_entropy'] for r in g
             if r['prec_entropy'] is not None]
        s_m = np.mean(s) if s else float('nan')
        print(f"  mult={mult} n={len(g):3d}: "
              f"energy={e:.3f} entropy={s_m:.3f} "
              f"improv={s_m-e:+.3f}")

    print(f"\nBy ligand:")
    for lig in ['Cl','Br','F']:
        g = [r for r in results if r['ligand']==lig]
        if not g: continue
        e = np.mean([r['prec_energy'] for r in g])
        s = [r['prec_entropy'] for r in g
             if r['prec_entropy'] is not None]
        s_m = np.mean(s) if s else float('nan')
        print(f"  {lig}: n={len(g):3d} "
              f"energy={e:.3f} entropy={s_m:.3f} "
              f"improv={s_m-e:+.3f}")
