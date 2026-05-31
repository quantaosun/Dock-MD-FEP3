#!/usr/bin/env python3
"""
Prepare protein + ligand inputs for OpenFE ABFE from any PDB structure.

Adapted from: Protein_ligand_Input_and_per-residue_decomp_20231018.ipynb

Usage
-----
  python 01_prepare_inputs.py --pdb <PDB_ID> --resname <LIGAND_RESIDUE>

Examples
--------
  python 01_prepare_inputs.py --pdb 6I5I  --resname H3E  --charge 0
  python 01_prepare_inputs.py --pdb 181L  --resname TMP  --charge 0

Produces:
  receptor.pdb     — protein only, with hydrogens (for 02_run_abfe_openfe.py)
  ligand.sdf       — ligand with correct element types (for 02_run_abfe_openfe.py)
  ligand.mol2      — ligand with GAFF2 atom types and AM1-BCC charges
  starting_end.pdb — copy of receptor.pdb (compatibility)
  complex.pdb      — receptor + ligand merged (visualisation)
"""

import os, sys, subprocess, tempfile, shutil, argparse, textwrap
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

# ── config ──────────────────────────────────────────────────────────
PDB_URL  = "https://files.rcsb.org/download/{}.pdb"
WORK_DIR = Path.cwd()


def log(msg):
    print(f"  {msg}")


def download_pdb(pdb_id: str) -> Path:
    url = PDB_URL.format(pdb_id)
    dest = WORK_DIR / f"{pdb_id}.pdb"
    if dest.exists():
        log(f"  Using cached {dest}")
        return dest
    import urllib.request
    log(f"  Downloading {url} …")
    urllib.request.urlretrieve(url, dest)
    log(f"  Saved to {dest}")
    return dest


def extract_ligand_pdb(pdb_path: Path, resname: str) -> Path:
    """Extract HETATM records for *resname* into a separate PDB file."""
    lig_pdb = WORK_DIR / f"ligand_raw.pdb"
    count = 0
    with open(pdb_path) as fin, open(lig_pdb, "w") as fout:
        for line in fin:
            if line.startswith("HETATM") and line[17:20].strip() == resname:
                # Rewrite as ATOM record (some tools prefer this)
                fout.write("ATOM  " + line[6:])
                count += 1
    log(f"  Extracted {count} atoms for ligand '{resname}' → {lig_pdb}")
    if count == 0:
        raise RuntimeError(f"Ligand {resname} not found in {pdb_path}")
    return lig_pdb


def extract_protein_pdb(pdb_path: Path) -> Path:
    """Extract only ATOM records (protein), strip waters and heteroatoms."""
    prot_pdb = WORK_DIR / "receptor_raw.pdb"
    count = 0
    with open(pdb_path) as fin, open(prot_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM"):
                fout.write(line)
                count += 1
            elif line.startswith("TER"):
                fout.write(line)
            elif line.startswith("END"):
                fout.write(line)
    log(f"  Extracted {count} protein ATOM records → {prot_pdb}")
    return prot_pdb


def prepare_ligand(raw_pdb: Path, net_charge: int) -> tuple[Path, Path]:
    """
    Ligand preparation pipeline:
      1. OpenBabel: PDB → MOL (preserves bonding)
      2. RDKit: add H, minimize (heavy atoms restrained), compute Gasteiger charges
      3. pdb4amber: clean up
      4. antechamber: GAFF2 atom types + AM1-BCC charges → ligand.mol2
      5. parmchk2: check missing parameters

    Also writes ligand.sdf with proper element types for OpenFE.

    Returns (mol2_path, frcmod_path).
    """
    # ── Step 1: PDB → MOL via OpenBabel ─────────────────────────────
    log("  Step 1/5: OpenBabel PDB → MOL")
    mol = next(pybel.readfile("pdb", str(raw_pdb)))
    mol.write("mol", str(WORK_DIR / "ligand_temp.mol"), overwrite=True)

    # ── Step 2: RDKit add H + minimize ──────────────────────────────
    log("  Step 2/5: RDKit add H, minimize (heavy atoms restrained)")
    rdmol = Chem.MolFromMolFile(str(WORK_DIR / "ligand_temp.mol"), removeHs=True)
    if rdmol is None:
        raise RuntimeError("RDKit could not parse ligand from MOL")

    hmol = Chem.AddHs(rdmol)
    mp = AllChem.MMFFGetMoleculeProperties(hmol)
    if mp is None:
        raise RuntimeError("MMFF cannot parameterize this ligand")
    ff = AllChem.MMFFGetMoleculeForceField(hmol, mp)
    for a in hmol.GetAtoms():
        if a.GetAtomicNum() > 1:
            ff.MMFFAddPositionConstraint(a.GetIdx(), 0.0, 1.0e4)
    ff.Minimize(maxIts=2000)
    charge = Chem.GetFormalCharge(hmol)
    log(f"    Net charge (from RDKit): {charge}")

    lig_h_pdb = WORK_DIR / "ligand_H.pdb"
    Chem.MolToPDBFile(hmol, str(lig_h_pdb))
    log(f"    Written {lig_h_pdb}")

    # Save clean SDF for OpenFE (correct element types)
    lig_sdf = WORK_DIR / "ligand.sdf"
    Chem.MolToMolFile(hmol, str(lig_sdf))
    log(f"    Written {lig_sdf}")

    # ── Step 3: pdb4amber ─────────────────────────────────────────────
    log("  Step 3/5: pdb4amber")
    lig_h_clean = WORK_DIR / "ligand_h.pdb"
    subprocess.run(
        ["pdb4amber", "-i", str(lig_h_pdb), "-o", str(lig_h_clean)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    if not lig_h_clean.exists() or os.path.getsize(lig_h_clean) == 0:
        shutil.copy(lig_h_pdb, lig_h_clean)
        log("    pdb4amber produced empty output, using original")

    # ── Step 4: antechamber → MOL2 with GAFF2 + AM1-BCC ─────────────
    log("  Step 4/5: antechamber (GAFF2, AM1-BCC)")
    lig_mol2 = WORK_DIR / "ligand.mol2"
    subprocess.run(
        ["antechamber",
         "-i", str(lig_h_clean),  "-fi", "pdb",
         "-o", str(lig_mol2),     "-fo", "mol2",
         "-c", "bcc",
         "-nc", str(net_charge),
         "-rn", "LIG",
         "-at", "gaff2",
         "-pf", "y"],
        capture_output=True, text=True, check=True, timeout=300,
    )
    if not lig_mol2.exists():
        raise RuntimeError("antechamber failed — no ligand.mol2 produced")

    # ── Step 5: parmchk2 ─────────────────────────────────────────────
    log("  Step 5/5: parmchk2")
    lig_frcmod = WORK_DIR / "ligand.frcmod"
    subprocess.run(
        ["parmchk2", "-i", str(lig_mol2), "-f", "mol2",
         "-o", str(lig_frcmod), "-s", "gaff2"],
        capture_output=True, text=True, check=True, timeout=120,
    )

    # Clean up intermediates
    for f in ["ligand_temp.mol", "ligand_H.pdb", "ligand_h.pdb"]:
        (WORK_DIR / f).unlink(missing_ok=True)

    log(f"  ✓ ligand.mol2  (GAFF2 types, AM1-BCC charges)")
    log(f"  ✓ ligand.sdf   (for OpenFE — proper element names)")
    return lig_mol2, lig_frcmod


def prepare_protein(raw_pdb: Path) -> Path:
    """
    Protein preparation using the notebook's approach:
      1. cpptraj prepareforleap → strip waters, no H
      2. pdb4amber → clean (resolve alt-locs, add TER cards)

    Returns starting_end.pdb path.
    """
    log("  Step 1/3: cpptraj prepareforleap")
    prep_in = WORK_DIR / "prepareforleap.in"
    prep_in.write_text(textwrap.dedent(f"""\
        parm {raw_pdb}
        loadcrd {raw_pdb} name edited
        prepareforleap crdset edited name from-prepareforleap \\
            pdbout {WORK_DIR / "starting1.pdb"} nowat noh
        go
    """))
    subprocess.run(
        ["cpptraj", "-i", str(prep_in)],
        capture_output=True, text=True, check=True, timeout=120,
    )

    log("  Step 2/3: pdb4amber")
    starting_end = WORK_DIR / "starting_end.pdb"
    subprocess.run(
        ["pdb4amber", "-i", str(WORK_DIR / "starting1.pdb"),
         "-o", str(starting_end), "-a"],
        capture_output=True, text=True, check=True, timeout=120,
    )

    # Clean up
    for f in ["prepareforleap.in", "starting1.pdb", "starting1.pdb",
              prepareforleap_in := "prepareforleap.in"]:
        (WORK_DIR / f).unlink(missing_ok=True)

    if not starting_end.exists():
        # Fallback: just use the raw protein with pdb4amber
        log("  prepareforleap failed, falling back to raw pdb4amber")
        raw_clean = WORK_DIR / "receptor.pdb"
        subprocess.run(
            ["pdb4amber", "-i", str(raw_pdb), "-o", str(raw_clean),
             "--dry", "--most-populous"],
            capture_output=True, text=True, timeout=120,
        )
        starting_end = raw_clean

    log(f"  ✓ {starting_end}")
    return starting_end


def build_receptor(starting_end: Path):
    """
    Build a clean receptor.pdb (protein only, with hydrogens).

    The protein from cpptraj + pdb4amber has no H atoms, which causes
    ff14SB template matching to fail downstream.  We use PDBFixer to
    add missing atoms and hydrogens at pH 7.0.

    Outputs:
      receptor.pdb    — clean protein with H (for ProteinComponent)
      starting_end.pdb— same as receptor.pdb (name expected by 02_run_abfe_openfe.py)
      complex.pdb     — protein + ligand merged (visualisation only)
    """
    log("\n  Adding hydrogens to protein with PDBFixer …")
    import pdbfixer
    from openmm.app import PDBFile

    fixer = pdbfixer.PDBFixer(filename=str(starting_end))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    receptor = WORK_DIR / "receptor.pdb"
    with open(receptor, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    n_atoms = sum(1 for _ in open(receptor) if _.startswith("ATOM"))
    log(f"  ✓ {receptor}  ({n_atoms} protein atoms)")

    # Copy to starting_end.pdb (the file name expected by 02_run_abfe_openfe.py)
    shutil.copy(receptor, WORK_DIR / "starting_end.pdb")

    # Also produce a merged PDB with ligand for visualisation
    lig_raw = WORK_DIR / "ligand_raw.pdb"
    merged = WORK_DIR / "complex.pdb"
    with open(receptor) as f:
        prot_lines = [l for l in f if l.startswith(("ATOM", "HETATM", "TER", "END"))]
    with open(lig_raw) as f:
        lig_lines = [l for l in f if l.startswith(("ATOM", "HETATM"))]

    with open(merged, "w") as f:
        f.write("REMARK  Prepared for OpenFE ABFE | PDB complex\n")
        for line in prot_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                f.write(line)
            elif line.startswith("END") or line.startswith("TER"):
                f.write(line)
        f.write("TER\n")
        for line in lig_lines:
            # Rename residue to LIG for clarity
            f.write(line[:17] + "LIG" + line[20:])
        f.write("TER\nEND\n")
    log(f"  ✓ {merged}  (protein + ligand, visualisation only)")


def main():
    parser = argparse.ArgumentParser(description="Prepare inputs for OpenFE ABFE")
    parser.add_argument("--pdb", default="6I5I", help="PDB ID (default: 6I5I)")
    parser.add_argument("--charge", type=int, default=0, help="Ligand net charge (default: 0)")
    parser.add_argument("--resname", default="H3E", help="Ligand residue name in PDB (default: H3E)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Preparing inputs for PDB {args.pdb}")
    print(f"  Ligand: {args.resname}, charge={args.charge}")
    print(f"{'='*60}\n")

    # 1. Download
    print("── Step 1: Download PDB ──")
    pdb_path = download_pdb(args.pdb)

    # 2. Extract components
    print("\n── Step 2: Extract components ──")
    lig_raw_pdb = extract_ligand_pdb(pdb_path, args.resname)
    prot_raw_pdb = extract_protein_pdb(pdb_path)

    # 3. Prepare ligand
    print("\n── Step 3: Prepare ligand (GAFF2 + AM1-BCC) ──")
    lig_mol2, lig_frcmod = prepare_ligand(lig_raw_pdb, args.charge)
    print(f"    ligand.mol2 : {lig_mol2}")
    print(f"    ligand.frcmod : {lig_frcmod}")

    # 4. Prepare protein
    print("\n── Step 4: Prepare protein ──")
    starting_end = prepare_protein(prot_raw_pdb)

    # 5. Build receptor (add H) + merged PDB
    print("\n── Step 5: Build receptor with hydrogens ──")
    build_receptor(starting_end)

    # Clean up temp intermediates (keep the raw downloaded PDB)
    for f in ["prepareforleap.in", "receptor_raw.pdb", "ligand_raw.pdb",
              "starting1.pdb", "ligand_temp.mol", "ligand_H.pdb", "ligand_h.pdb",
              f"{args.pdb}.pdb"]:
        (WORK_DIR / f).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"  Done! Input files ready for OpenFE ABFE:")
    print(f"    receptor.pdb      — protein only (with H)")
    print(f"    ligand.mol2       — ligand (GAFF2, AM1-BCC charges)")
    print(f"    starting_end.pdb  — merged (for visualisation)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
