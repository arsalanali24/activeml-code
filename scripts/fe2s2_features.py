import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_INTERFACE_LAYER"] = "LP64"
import ctypes
ctypes.CDLL("/pc2/users/h/hpcmual/activeml/lib/lapack_wrapper.so")
"""
UHF + MP2 + DMRG features for Fe2S2 systems.
Equivalent of pyscf_features_v2.py for dinuclear Fe-S.
Usage: python fe2s2_features.py <input_json>
"""
import sys, os, json, time, logging
import numpy as np
from pyscf import gto, scf, mp, mcscf
from pyblock2.dmrgscf import DMRGCI

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ── LOAD INPUT ────────────────────────────────────────────────
inp_path = sys.argv[1]
inp      = json.load(open(inp_path))
name     = inp['name']

outdir   = os.path.expanduser(
    '~/activeml/data/fe2s2_features')
os.makedirs(outdir, exist_ok=True)
outfile  = f'{outdir}/{name}.json'

if os.path.exists(outfile):
    log.info(f'SKIP: {name}')
    sys.exit(0)

charge   = inp['charge']
spin_bs  = inp['spin']
atom_str = inp['atom_str']
# Calculate valid high-spin dynamically
_mol_tmp = gto.Mole()
_mol_tmp.atom   = atom_str
_mol_tmp.charge = charge
# spin set after electron count
_mol_tmp.basis  = "def2-SVP"
_mol_tmp.verbose = 0
# determine correct spin for temp mol
_mol_tmp.spin = 0
try:
    _mol_tmp.build()
except RuntimeError:
    _mol_tmp.spin = 1
    _mol_tmp.build()
_n_elec = _mol_tmp.nelectron
spin_hs = min(10, _n_elec)
while spin_hs >= 0:
    if (_n_elec - spin_hs) >= 0 and (_n_elec - spin_hs) % 2 == 0:
        break
    spin_hs -= 1
if spin_hs < 0: spin_hs = 0

result = {
    'name'      : name,
    'cluster'   : 'Fe2S2',
    'terminal'  : inp['terminal'],
    'charge'    : charge,
    'spin'      : spin_bs,
    'mult'      : spin_bs + 1,
    'fe_fe_dist': inp['fe_fe_dist'],
    'distort'   : inp['distort'],
}

try:
    # ── STEP 1: BUILD MOLECULE ────────────────────────────────
    mol_hs        = gto.Mole()
    mol_hs.atom   = atom_str
    mol_hs.charge = charge
    mol_hs.spin   = spin_hs
    mol_hs.basis  = 'def2-SVP'
    mol_hs.verbose = 0
    mol_hs.build()

    result['n_electrons'] = mol_hs.nelectron
    result['n_basis']     = mol_hs.nao

    # ── STEP 2: HIGH-SPIN UHF ─────────────────────────────────
    t0    = time.time()
    mf_hs = scf.UHF(mol_hs)
    mf_hs.max_cycle = 500
    mf_hs.conv_tol  = 1e-9
    mf_hs.verbose   = 0
    for damp, shift in [(0.0,0.0),(0.3,0.2),(0.5,0.5)]:
        mf_hs.damp        = damp
        mf_hs.level_shift = shift
        mf_hs.kernel()
        if mf_hs.converged: break

    result['hs_uhf_converged'] = bool(mf_hs.converged)
    result['hs_uhf_energy']    = float(mf_hs.e_tot)
    result['hs_spin_contam']   = float(
        mf_hs.spin_square()[0])

    # ── STEP 3: BROKEN-SYMMETRY UHF ──────────────────────────
    mol_bs        = mol_hs.copy()
    mol_bs.spin   = spin_bs
    mol_bs.build()

    mf_bs = scf.UHF(mol_bs)
    mf_bs.max_cycle = 800
    mf_bs.conv_tol  = 1e-8
    mf_bs.verbose   = 0

    # BS guess: swap alpha/beta on Fe2
    dm_hs     = mf_hs.make_rdm1()
    ao_labels = mol_bs.ao_labels()
    fe2_aos   = [i for i,l in enumerate(ao_labels)
                 if l.split()[0] == '1']
    dm_a = dm_hs[0].copy()
    dm_b = dm_hs[1].copy()
    dm_a_new = dm_a.copy()
    dm_b_new = dm_b.copy()
    for i in fe2_aos:
        for j in fe2_aos:
            dm_a_new[i,j] = dm_b[i,j]
            dm_b_new[i,j] = dm_a[i,j]

    for damp, shift in [(0.0,0.0),(0.3,0.2),(0.5,0.5)]:
        mf_bs.damp        = damp
        mf_bs.level_shift = shift
        mf_bs.kernel((dm_a_new, dm_b_new))
        if mf_bs.converged: break

    s2_bs = float(mf_bs.spin_square()[0])
    result['bs_uhf_converged'] = bool(mf_bs.converged)
    result['bs_uhf_energy']    = float(mf_bs.e_tot)
    result['bs_spin_contam']   = s2_bs

    # ── STEP 4: EXTRACT HF FEATURES ──────────────────────────
    e_a, e_b   = mf_bs.mo_energy
    e_mean     = (e_a + e_b) / 2
    occ_a, occ_b = mf_bs.mo_occ
    occ_total  = occ_a + occ_b

    occ_idx  = np.where(occ_total > 0.5)[0]
    virt_idx = np.where(occ_total <= 0.5)[0]
    homo_e   = float(e_mean[occ_idx[-1]])
    lumo_e   = float(e_mean[virt_idx[0]])

    result['homo_e']       = homo_e
    result['lumo_e']       = lumo_e
    result['homo_lumo_gap']= lumo_e - homo_e
    result['n_occ']        = int(len(occ_idx))
    result['n_virt']       = int(len(virt_idx))

    # Fe-Fe distance (from geometry)
    result['fe_fe_dist_actual'] = inp['fe_fe_dist']

    # HS - BS energy difference (J coupling proxy)
    delta_E = float(mf_hs.e_tot - mf_bs.e_tot)
    result['delta_E_HS_BS']  = delta_E
    result['J_approx_meV']   = float(
        delta_E / (2 * (spin_hs/2) *
                   (spin_hs/2 + 1)) * 27211)

    # ── STEP 5: MP2 FEATURES ─────────────────────────────────
    try:
        mp2 = mp.MP2(mf_bs)
        mp2.verbose = 0
        mp2.run()
        result['mp2_corr']   = float(mp2.e_corr)
        result['largest_t2'] = float(
            np.abs(mp2.t2).max()
            if not isinstance(mp2.t2, tuple)
            else max(np.abs(t).max()
                     for t in mp2.t2
                     if t is not None))

        # MP2 natural orbital occupations
        dm_mp2 = mp2.make_rdm1()
        if isinstance(dm_mp2, tuple):
            dm_mp2 = dm_mp2[0] + dm_mp2[1]
        S_ao   = mol_bs.intor('int1e_ovlp')
        X      = np.linalg.inv(
            np.linalg.cholesky(S_ao)).T
        dm_orth = X @ dm_mp2 @ X.T
        mp2_occ = np.sort(
            np.linalg.eigvalsh(dm_orth))[::-1]
        result['n_frac_mp2_010'] = int(np.sum(
            (mp2_occ > 0.10) & (mp2_occ < 1.90)))
        result['n_frac_mp2_020'] = int(np.sum(
            (mp2_occ > 0.20) & (mp2_occ < 1.80)))
        result['mp2_ok'] = True
    except Exception as e:
        log.warning(f'MP2 failed {name}: {e}')
        result['mp2_corr']       = None
        result['largest_t2']     = None
        result['n_frac_mp2_010'] = None
        result['n_frac_mp2_020'] = None
        result['mp2_ok']         = False

    # ── STEP 6: DMRG ACTIVE SPACE ────────────────────────────
    window20 = sorted(
        list(occ_idx[-10:]) + list(virt_idx[:10]))

    n_act_e = 20
    for n in [20, 18, 22, 16, 24]:
        if ((mol_bs.nelectron - n) >= 0 and
            (mol_bs.nelectron - n) % 2 == 0):
            n_act_e = n
            break

    try:
        mo_avg = ((mf_bs.mo_coeff[0] +
                   mf_bs.mo_coeff[1]) / 2)
        mc = mcscf.CASSCF(mf_bs, 20, n_act_e)
        mc.fcisolver = DMRGCI(mf_bs)
        mc.fcisolver.dmrg_args = {"startM": 250, "maxM": 500, "schedule": "default", "sweep_tol": 1e-6, "memory": 60000}
        mc.max_cycle_macro     = 20
        mc.conv_tol            = 1e-6
        mc.verbose             = 0
        mo = mc.sort_mo(window20,
                        mo_coeff=mo_avg, base=0)
        mc.kernel(mo)

        casdm1    = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_cas, _ = np.linalg.eigh(casdm1)
        no_cas    = np.sort(no_cas)[::-1]
        eps       = 1e-12
        n_clip    = np.clip(no_cas/2, eps, 1-eps)
        s_i       = -(n_clip*np.log(n_clip) +
                      (1-n_clip)*np.log(1-n_clip))
        frac      = int(np.sum(
            (no_cas > 0.02) & (no_cas < 1.98)))

        result['dmrg_converged']  = bool(mc.converged)
        result['dmrg_energy']     = float(mc.e_tot)
        result['no_occ']          = no_cas.tolist()
        result['s_i']             = s_i.tolist()
        result['n_active']        = frac
        result['dmrg_ok']         = True

        log.info(f'{name}: n_active={frac} '
                 f'J={result["J_approx_meV"]:.1f}meV '
                 f'gap={result["homo_lumo_gap"]:.3f}')

    except Exception as e:
        log.warning(f'DMRG failed {name}: {e}')
        result['dmrg_ok']     = False
        result['n_active']    = None
        result['dmrg_energy'] = None
        result['no_occ']      = None
        result['s_i']         = None

    result['status']     = 'done'
    result['time_total'] = round(time.time()-t0, 1)

except Exception as e:
    log.error(f'FAILED {name}: {e}')
    result['status'] = f'error: {e}'

json.dump(result, open(outfile,'w'), indent=2)
sys.exit(0)
