"""
Verify all systems before submission.
Checks parity, electron count, spin validity.
Run this BEFORE submitting any job array.
"""
import sys, os
from pyscf import gto

METAL_Z = {'Fe':26,'Mn':25,'Cr':24,'Co':27,
           'Ni':28,'Cu':29,'Zn':30,'Ti':22,'V':23}

LIG_Z   = {'Cl':17,'Br':35,'F':9,'I':53,
           'N':7,'O':8,'S':16,'C':6,'H':1,
           'P':15}

LIG_CHARGE = {'Cl':-1,'Br':-1,'F':-1,'I':-1,
              'N':-3,'O':-2,'S':-2,'C':-4,
              'H':-1,'P':-3}

def n_electrons(metal, ligand, n_lig, charge):
    z_m = METAL_Z.get(metal, 26)
    z_l = LIG_Z.get(ligand, 8)
    return z_m + n_lig * z_l - charge

def parity_ok(metal, ligand, n_lig, charge, spin):
    n_elec = n_electrons(metal, ligand, n_lig, charge)
    return (n_elec % 2) == (spin % 2), n_elec

def max_spin(metal, ligand, n_lig, charge):
    """Maximum physically reasonable spin."""
    n_elec, _ = parity_ok(metal, ligand, n_lig, charge, 0)
    n_elec = n_electrons(metal, ligand, n_lig, charge)
    # Rough upper limit based on d-electrons
    # Conservative: max spin = 5 (sextet, half-filled d5)
    return min(5, n_elec % 2 + 4)

def verify_system(metal, ligand, n_lig, charge, spin):
    """Returns (valid, reason)."""
    n_elec = n_electrons(metal, ligand, n_lig, charge)

    # Check parity
    if (n_elec % 2) != (spin % 2):
        return False, f"parity: n_elec={n_elec} spin={spin}"

    # Check reasonable electron count
    if n_elec < 10:
        return False, f"too few electrons: {n_elec}"
    if n_elec > 400:
        return False, f"too many electrons: {n_elec}"

    # Check spin is non-negative
    if spin < 0:
        return False, f"negative spin: {spin}"

    # Check spin is not absurdly large
    if spin > 10:
        return False, f"spin too large: {spin}"

    return True, "ok"

if __name__ == "__main__":
    # Test a few systems
    test_cases = [
        # (metal, ligand, n_lig, charge, spin, expected)
        ('Fe', 'Cl', 4, -2, 4, True),   # quintet ✓
        ('Fe', 'Cl', 4, -2, 3, False),  # parity ✗
        ('Fe', 'N',  4, -2, 4, True),   # N ligand ✓
        ('Co', 'O',  4, -2, 3, True),   # O ligand ✓
        ('Mn', 'S',  6, -2, 5, True),   # S ligand ✓
        ('Cu', 'N',  4, -2, 1, True),   # Cu doublet ✓
        ('Fe', 'C',  6, 0,  4, True),   # C ligand ✓
    ]

    print("=== VERIFICATION TESTS ===\n")
    all_pass = True
    for metal,lig,n_lig,charge,spin,expected in test_cases:
        valid, reason = verify_system(
            metal, lig, n_lig, charge, spin)
        n_elec = n_electrons(metal, lig, n_lig, charge)
        status = "✓" if valid == expected else "✗ WRONG"
        print(f"{metal}_{lig}{n_lig}_chg{charge}_spin{spin}: "
              f"n_elec={n_elec} valid={valid} "
              f"({reason}) {status}")
        if valid != expected:
            all_pass = False

    print(f"\nAll tests passed: {all_pass}")
