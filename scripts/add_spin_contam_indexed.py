"""
add_spin_contam_indexed.py
Reads file list from spinc_missing.txt — stable chunk boundaries.
Usage: python add_spin_contam_indexed.py <chunk_index> <chunk_total>
"""
import sys, os, json, logging
import numpy as np
from pyscf import gto, scf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

INDEX = os.path.expanduser('~/activeml/scripts/spinc_missing.txt')

def rebuild_mol(d):
    metal  = d['metal']
    ligand = d['ligand']
    n_lig  = d['n_ligands']
    charge = d['charge']
    spin   = d['spin']
    dist   = d.get('dist_ang', 2.1)
    geom   = d.get('geometry', 'oct' if n_lig == 6 else 'tet')

    if geom == 'csd_real':
        return None
    if n_lig == 6 or geom == 'oct':
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),(0,0,dist),(0,0,-dist)]
    elif geom in ('sq_pl', 'square_planar') or (n_lig == 4 and 'sq' in geom):
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)]
    elif n_lig == 5:
        import math
        if 'tbp' in geom:
            pos = [(dist,0,0),
                   (-dist*0.5, dist*math.sqrt(3)/2, 0),
                   (-dist*0.5,-dist*math.sqrt(3)/2, 0),
                   (0,0,dist),(0,0,-dist)]
        else:
            pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),(0,0,dist)]
    elif n_lig == 4:
        s = dist / np.sqrt(3)
        pos = [(s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)]
    else:
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),(0,0,dist),(0,0,-dist)]

    atom_str = f"{metal} 0 0 0\n"
    for p in pos[:n_lig]:
        atom_str += f"{ligand} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n"

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = 'def2-svp'
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.max_memory = 8000
    mol.build()
    return mol

def process(filepath):
    try:
        d = json.load(open(filepath))
    except:
        return False

    if d.get('status') != 'ok':
        return True
    if 'spin_contamination' in d:
        return True

    geom = d.get('geometry', '')
    if geom == 'csd_real':
        d['spin_contamination'] = 0.0
        json.dump(d, open(filepath, 'w'), indent=2)
        return True

    try:
        mol = rebuild_mol(d)
        if mol is None:
            return True
    except Exception as e:
        log.warning(f"Build failed {d.get('name','?')}: {e}")
        return False

    try:
        mf = scf.UHF(mol)
        mf.max_cycle = 200
        mf.conv_tol  = 1e-8
        mf.kernel()
        if not mf.converged:
            mf.damp = 0.3
            mf.kernel()

        S2, _ = mf.spin_square()
        S = d['spin'] / 2.0
        spin_contam = float(S2 - S * (S + 1))

        d['spin_contamination'] = spin_contam
        json.dump(d, open(filepath, 'w'), indent=2)
        log.info(f"  {d['name']}: sc={spin_contam:.4f}")
        return True
    except Exception as e:
        log.warning(f"UHF failed {d.get('name','?')}: {e}")
        return False

if __name__ == '__main__':
    chunk_idx   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunk_total = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    all_files = [l.strip() for l in open(INDEX) if l.strip()]
    chunk_size = len(all_files) // chunk_total + 1
    start = chunk_idx * chunk_size
    end   = min(start + chunk_size, len(all_files))
    files = all_files[start:end]

    log.info(f"Chunk {chunk_idx}/{chunk_total}: {len(files)} files "
             f"(index {start}-{end})")
    done = 0
    for f in files:
        if process(f):
            done += 1
    log.info(f"Done: {done}/{len(files)}")
