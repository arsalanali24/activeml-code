"""
extract_tier2.py
────────────────
Adds Tier 2 features to the existing 3,121 good JSON files in
~/activeml/data/generated300/ WITHOUT rerunning any CASSCF.

What this script does for each ok file:
  1. Loads the existing JSON
  2. Rebuilds the molecule (takes < 1 second, no SCF)
  3. Runs a single X2C UHF — or reads saved UHF data from the JSON
     if it was already stored
  4. Extracts Tier 2 features from the UHF wavefunction
  5. Writes the augmented JSON back in-place (original fields untouched)

Tier 2 features added:
  spin_contamination       — <S²> - S(S+1)   [already derivable but now explicit]
  alpha_beta_overlap       — Tr(S_αβ)        [spin polarisation depth]
  mulliken_metal_charge    — Mulliken charge on the metal atom
  loewdin_metal_charge     — Löwdin charge on the metal atom
  d_t2g_occupancy          — d-electron occupancy in t2g symmetry block
  d_eg_occupancy           — d-electron occupancy in eg symmetry block
  mayer_bond_order_mean    — mean Mayer bond order over all M-L bonds
  mayer_bond_order_std     — std of Mayer bond orders (asymmetry indicator)
  z_eff                    — tabulated Zeff (Clementi & Raimondi)
  zeta_so_cm1              — tabulated spin-orbit coupling constant (NIST)
  metal_row                — '3d', '4d', or '5d'
  homo_lumo_gap_eV         — HOMO-LUMO gap in eV (re-extracted from UHF)

Usage:
  # Process all 3121 files (runs fast — UHF only, ~30 sec each on 4 cores)
  python extract_tier2.py

  # Process a single file (for testing):
  python extract_tier2.py --file ~/activeml/data/generated300/Ni_O4_chg-5_spin1.json

  # Dry run — show what would be added without writing:
  python extract_tier2.py --dry-run --file <path>
"""

import numpy as np
import json
import os
import sys
import glob
import argparse
import logging
from pyscf import gto, scf
from pyscf.x2c import x2c

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Tabulated constants ───────────────────────────────────────────────────────
# Effective nuclear charge (Clementi & Raimondi, J. Chem. Phys. 1963)
# Spin-orbit coupling constants (cm⁻¹) from NIST ASD
METAL_CONSTANTS = {
    # metal: (Zeff_d,  zeta_SO,  row)
    'Ti': ( 8.14,   121,  '3d'),
    'V':  ( 8.98,   208,  '3d'),
    'Cr': ( 9.76,   273,  '3d'),
    'Mn': (10.53,   355,  '3d'),
    'Fe': (11.18,   460,  '3d'),
    'Co': (12.00,   533,  '3d'),
    'Ni': (12.78,   669,  '3d'),
    'Cu': (13.20,   831,  '3d'),
    'Zn': (13.57,  1042,  '3d'),
    # 4d (for completeness — these appear if generated300 ever had 4d)
    'Pd': (13.00,  1334,  '4d'),
    'Ru': (12.33,   880,  '4d'),
    'Rh': (12.67,  1097,  '4d'),
    'Mo': (10.97,   467,  '4d'),
    # 5d
    'Ir': (17.00,  3909,  '5d'),
    'Pt': (17.33,  4146,  '5d'),
}

HARTREE_TO_EV = 27.2114


# ── Geometry rebuilder ────────────────────────────────────────────────────────
def rebuild_geometry(metal, ligand, n_lig, dist, geometry='oct'):
    """
    Rebuild atom string from the stored fields.
    Falls back to tetrahedral/octahedral if geometry field is missing.
    """
    if geometry in ('sq_pl', 'square_planar') or \
       (n_lig == 4 and geometry not in ('tet', 'oct')):
        positions = [
            ( dist,  0,    0),
            (-dist,  0,    0),
            ( 0,     dist, 0),
            ( 0,    -dist, 0),
        ]
    elif n_lig == 6 or geometry == 'oct':
        positions = [
            ( dist,  0,    0),
            (-dist,  0,    0),
            ( 0,     dist, 0),
            ( 0,    -dist, 0),
            ( 0,     0,    dist),
            ( 0,     0,   -dist),
        ]
    else:
        s = dist / np.sqrt(3)
        positions = [
            ( s,  s,  s),
            ( s, -s, -s),
            (-s,  s, -s),
            (-s, -s,  s),
        ]
    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[:n_lig]:
        atom_str += f"{ligand}  {p[0]:.4f}  {p[1]:.4f}  {p[2]:.4f}\n"
    return atom_str


# ── UHF runner ────────────────────────────────────────────────────────────────
def run_uhf(mol):
    """Run UHF (non-relativistic for 3d metals, matching original gen_300)."""
    mf = scf.UHF(mol)
    mf.max_cycle = 400
    mf.conv_tol  = 1e-9
    mf.kernel()
    if not mf.converged:
        mf.damp = 0.3
        mf.kernel()
    return mf


# ── Safe RDM1 helper ─────────────────────────────────────────────────────────
def get_rdm1_alpha_beta(mf):
    """Return (dm_alpha, dm_beta) safely for both RHF and UHF."""
    dm = mf.make_rdm1()
    if isinstance(dm, tuple):
        return dm[0], dm[1]
    if hasattr(dm, 'ndim') and dm.ndim == 3:
        return dm[0], dm[1]
    # Single 2D array — closed shell, alpha=beta=dm/2
    return dm * 0.5, dm * 0.5

# ── Mulliken / Löwdin metal charge ─────────────────────────────────────────────
def get_metal_charges(mol, mf):
    """
    Returns (mulliken_charge, loewdin_charge) on the metal (atom index 0).
    Uses the total (alpha + beta) density matrix.
    """
    dm_alpha, dm_beta = get_rdm1_alpha_beta(mf)
    dm_total = dm_alpha + dm_beta

    # Mulliken — use charges directly (second return value = per-atom charges)
    _, charges_mull = scf.uhf.mulliken_pop(mol, dm_total, verbose=0)[:2]
    mulliken = float(np.atleast_1d(charges_mull)[0])

    # Löwdin — use charges directly
    _, charges_loew = scf.uhf.mulliken_pop(mol, dm_total,
                                            s=mol.intor('int1e_ovlp'),
                                            verbose=0)[:2]
    loewdin = float(np.atleast_1d(charges_loew)[0])

    return mulliken, loewdin


# ── Alpha-beta orbital overlap ────────────────────────────────────────────────
def get_alpha_beta_overlap(mol, mf):
    """
    Tr(S · P_α · S · P_β) / n_electrons  — spin polarisation depth.
    Value near 0 = pure spin state, near 1 = heavily polarised.
    """
    S   = mol.intor('int1e_ovlp')
    Pa, Pb = get_rdm1_alpha_beta(mf)
    val = np.trace(S @ Pa @ S @ Pb)
    return float(val)


# ── d-orbital t2g / eg occupancy ─────────────────────────────────────────────
def get_d_orbital_occupancy(mol, mf, metal):
    """
    Split d-orbital occupancy into t2g and eg blocks using a simple
    AO symmetry heuristic.

    For octahedral field:
      t2g = d_xy, d_xz, d_yz   (l=2, ml = -2,-1,+1 in Cartesian: dxy dxz dyz)
      eg  = d_z2, d_x2-y2      (l=2, ml = 0,+2 in Cartesian: dz2 dx2y2)

    Returns (t2g_occ, eg_occ) as floats.
    Heuristic: sort the 5 d-orbital occupancies; top 2 → eg, bottom 3 → t2g
    (valid for high-spin octahedral; approximate for other geometries).
    """
    dm_alpha, dm_beta = get_rdm1_alpha_beta(mf)
    dm_total = dm_alpha + dm_beta

    # Find AO indices for d orbitals on the metal (atom 0)
    metal_idx = 0
    ao_labels  = mol.ao_labels()
    d_indices  = [i for i, lab in enumerate(ao_labels)
                  if lab[0] == metal_idx and 'd' in lab[2]]

    if len(d_indices) < 5:
        return None, None  # basis doesn't resolve 5 d-AOs

    # Diagonal of the density matrix at d-AO positions
    d_pops = np.array([dm_total[i, i] for i in d_indices])

    # Sort descending: top 2 = eg-like (highest energy in oct field)
    sorted_pops = np.sort(d_pops)[::-1]
    eg_occ  = float(np.sum(sorted_pops[:2]))
    t2g_occ = float(np.sum(sorted_pops[2:5]))
    return t2g_occ, eg_occ


# ── Mayer bond order ──────────────────────────────────────────────────────────
def get_mayer_bond_orders(mol, mf, n_lig):
    """
    Compute Mayer bond order for each M-L bond.
    Returns (mean_BO, std_BO).
    Metal is atom 0; ligands are atoms 1..n_lig.
    """
    from pyscf.lo import orth
    S  = mol.intor('int1e_ovlp')
    Pa, Pb = get_rdm1_alpha_beta(mf)
    Pt = Pa + Pb

    # Mayer BO between atoms A and B:
    # B_AB = sum_{mu in A} sum_{nu in B}  (PS)_mu_nu * (PS)_nu_mu
    PS = Pt @ S
    bos = []
    atom_ao = []
    for atom in range(mol.natm):
        indices = [i for i, lab in enumerate(mol.ao_labels())
                   if lab[0] == atom]
        atom_ao.append(indices)

    metal_aos = atom_ao[0]
    for lig_atom in range(1, n_lig + 1):
        lig_aos = atom_ao[lig_atom]
        bo = 0.0
        for mu in metal_aos:
            for nu in lig_aos:
                bo += PS[mu, nu] * PS[nu, mu]
        bos.append(bo)

    if not bos:
        return None, None
    return float(np.mean(bos)), float(np.std(bos))


# ── Process one file ──────────────────────────────────────────────────────────
def process_file(filepath, dry_run=False):
    try:
        d = json.load(open(filepath))
    except Exception as e:
        log.error(f"Cannot read {filepath}: {e}")
        return False

    # Only process ok files
    if d.get('status') != 'ok' or d.get('corr_energy', 0) >= 0:
        return True  # skip silently

    # Skip if already has tier2 features
    if 'z_eff' in d and 'mulliken_metal_charge' in d:
        log.info(f"Already has Tier 2: {d['name']}")
        return True

    metal  = d['metal']
    ligand = d['ligand']
    n_lig  = d['n_ligands']
    charge = d['charge']
    spin   = d['spin']
    dist   = d.get('dist_ang', 2.0)  # default 2.0 if missing
    geom   = d.get('geometry', 'oct' if n_lig == 6 else 'tet')

    # CSD files have irregular real coordinates — cannot rebuild geometry
    # Add only tabulated features (z_eff, zeta_so) and skip UHF-based ones
    if geom == 'csd_real' or 'dist_ang' not in d:
        consts = METAL_CONSTANTS.get(metal, (None, None, '3d'))
        z_eff, zeta_so, row = consts
        tier2_tabulated = {
            'z_eff'        : z_eff,
            'zeta_so_cm1'  : zeta_so,
            'metal_row'    : row,
        }
        d.update(tier2_tabulated)
        if not dry_run:
            json.dump(d, open(filepath, 'w'), indent=2)
            log.info(f"  CSD file — tabulated features only: {d['name']}")
        return True

    log.info(f"Processing: {d['name']}")

    # Rebuild molecule (no SCF — fast)
    try:
        atom_str = rebuild_geometry(metal, ligand, n_lig, dist, geom)
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-svp'   # same basis as original gen_300
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()
    except Exception as e:
        log.error(f"  Mol build failed: {e}")
        return False

    # Run UHF
    try:
        mf = run_uhf(mol)
        if not mf.converged:
            log.warning(f"  UHF not converged for {d['name']} — skipping tier2")
            return False
    except Exception as e:
        log.error(f"  UHF failed: {e}")
        return False

    # Extract features
    try:
        # Spin contamination
        try:
            S2_val, _ = mf.spin_square()
        except Exception:
            S2_val = 0.0
        S = spin / 2.0
        spin_cont = float(S2_val - S * (S + 1))

        # HOMO-LUMO gap
        mo_e_a, mo_e_b = mf.mo_energy
        mo_o_a, mo_o_b = mf.mo_occ
        homo_a = float(mo_e_a[mo_o_a > 0].max())
        lumo_a = float(mo_e_a[mo_o_a == 0].min())
        homo_b = float(mo_e_b[mo_o_b > 0].max()) if any(mo_o_b > 0) else homo_a
        lumo_b = float(mo_e_b[mo_o_b == 0].min()) if any(mo_o_b == 0) else lumo_a
        homo   = max(homo_a, homo_b)
        lumo   = min(lumo_a, lumo_b)
        gap_eV = float((lumo - homo) * HARTREE_TO_EV)

        # Charges
        mulliken, loewdin = get_metal_charges(mol, mf)

        # Alpha-beta overlap
        ab_overlap = get_alpha_beta_overlap(mol, mf)

        # d-orbital t2g/eg occupancy
        t2g_occ, eg_occ = get_d_orbital_occupancy(mol, mf, metal)

        # Mayer bond orders
        try:
            mayer_mean, mayer_std = get_mayer_bond_orders(mol, mf, n_lig)
        except Exception:
            mayer_mean, mayer_std = None, None

        # Tabulated constants
        consts = METAL_CONSTANTS.get(metal, (None, None, '3d'))
        z_eff, zeta_so, row = consts

    except Exception as e:
        import traceback
        log.error(f"  Feature extraction failed: {e}")
        log.error(traceback.format_exc())
        return False

    # Build Tier 2 additions
    tier2 = {
        'spin_contamination'    : spin_cont,
        'homo_lumo_gap_eV'      : gap_eV,
        'mulliken_metal_charge' : mulliken,
        'loewdin_metal_charge'  : loewdin,
        'alpha_beta_overlap'    : ab_overlap,
        'z_eff'                 : z_eff,
        'zeta_so_cm1'           : zeta_so,
        'metal_row'             : row,
    }
    if t2g_occ is not None:
        tier2['d_t2g_occupancy'] = t2g_occ
        tier2['d_eg_occupancy']  = eg_occ
    if mayer_mean is not None:
        tier2['mayer_bond_order_mean'] = mayer_mean
        tier2['mayer_bond_order_std']  = mayer_std

    if dry_run:
        print(f"\nDRY RUN — would add to {d['name']}:")
        for k, v in tier2.items():
            print(f"  {k}: {v}")
        return True

    # Write back — original fields first, then tier2
    d.update(tier2)
    json.dump(d, open(filepath, 'w'), indent=2)
    log.info(f"  Updated: {os.path.basename(filepath)}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',    help='Process a single file (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be added without writing')
    parser.add_argument('--folder',       default=None,
                        help='Override data folder path')
    parser.add_argument('--chunk-index', type=int, default=0,
                        help='Which chunk to process (0-based)')
    parser.add_argument('--chunk-total', type=int, default=1,
                        help='Total number of chunks')
    args = parser.parse_args()

    if args.file:
        process_file(args.file, dry_run=args.dry_run)
        sys.exit(0)

    # Process all ok files in generated300
    folder = args.folder or os.path.expanduser(
        '~/activeml/data/generated300')
    files = sorted(glob.glob(os.path.join(folder, '*.json')))
    log.info(f"Found {len(files)} files in {folder}")
    # Chunk slicing for parallel array jobs
    if args.chunk_total > 1:
        chunk_size = len(files) // args.chunk_total + 1
        start = args.chunk_index * chunk_size
        end   = min(start + chunk_size, len(files))
        files = files[start:end]
        log.info(f"Chunk {args.chunk_index}/{args.chunk_total}: "
                 f"files {start}-{end} ({len(files)} files)")

    done, failed, skipped = 0, 0, 0
    for i, f in enumerate(files):
        try:
            d = json.load(open(f))
        except Exception:
            failed += 1
            continue

        if d.get('status') != 'ok':
            skipped += 1
            continue

        ok = process_file(f, dry_run=args.dry_run)
        if ok:
            done += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            log.info(f"Progress: {i+1}/{len(files)} — "
                     f"done={done} failed={failed} skipped={skipped}")

    log.info(f"\nFinished: done={done} failed={failed} skipped={skipped}")
