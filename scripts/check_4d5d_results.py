"""
check_4d5d_results.py
─────────────────────
Run after job_4d5d.sh completes to get a clean summary.
Usage:  python check_4d5d_results.py
"""
import json, glob, os
from collections import defaultdict

folders = {
    '3d (existing)': os.path.expanduser('~/activeml/data/generated300'),
    '4d/5d (new)':   os.path.expanduser('~/activeml/data/generated_4d5d'),
}

for label, folder in folders.items():
    if not os.path.isdir(folder):
        print(f"\n{label}: folder not found ({folder})")
        continue

    files = glob.glob(f'{folder}/*.json')
    by_status  = defaultdict(int)
    by_metal   = defaultdict(lambda: defaultdict(int))
    has_tier2  = 0

    for f in files:
        try:
            d = json.load(open(f))
            s = d.get('status', 'missing')
            m = d.get('metal', '?')
            by_status[s] += 1
            by_metal[m][s] += 1
            if 'z_eff' in d and 'mulliken_metal_charge' in d:
                has_tier2 += 1
        except Exception:
            by_status['corrupt'] += 1

    total = sum(by_status.values())
    ok    = by_status.get('ok', 0)
    print(f"\n{'='*50}")
    print(f"{label}:  {total} files,  {ok} ok  ({100*ok/max(total,1):.1f}%)")
    print(f"  Has Tier 2 features: {has_tier2}")
    print(f"  Status breakdown:")
    for k, v in sorted(by_status.items()):
        print(f"    {k}: {v}")
    print(f"  By metal (ok only):")
    for metal in sorted(by_metal.keys()):
        n_ok = by_metal[metal].get('ok', 0)
        n_fail = sum(v for k,v in by_metal[metal].items() if k != 'ok')
        print(f"    {metal}: {n_ok} ok, {n_fail} failed/other")
