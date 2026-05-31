#!/usr/bin/env python3
"""
Streamlit app — Prepare inputs for OpenFE ABFE from any PDB structure.

Usage
-----
  streamlit run app.py
"""

import streamlit as st
import subprocess, sys, os, shutil, tempfile, time, glob
from pathlib import Path

st.set_page_config(
    page_title="ABFE Input Preparer",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 ABFE Input Preparer")
st.markdown(
    """
Download a protein–ligand complex PDB from RCSB and prepare the two files needed
by `02_run_abfe_openfe.py`:
- **receptor.pdb** — protein with hydrogens
- **ligand.sdf** — ligand with correct element types

A **ligand.mol2** (GAFF2 / AM1-BCC) is also generated for reference.
"""
)

st.divider()

# ── Input form ──────────────────────────────────────────────────────────
with st.form("pdb_form"):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        pdb_id = st.text_input("PDB ID", value="6I5I",
                               help="Four-character PDB identifier, e.g. 6I5I, 3HTB, 181L")
    with col2:
        resname = st.text_input("Ligand residue", value="H3E",
                                help="HETATM residue name of your ligand (3-4 chars)")
    with col3:
        charge = st.number_input("Net charge", value=0, step=1,
                                 help="Ligand net formal charge (usually 0)")

    submitted = st.form_submit_button("🚀 Prepare inputs", type="primary",
                                      use_container_width=True)

# ── Processing ──────────────────────────────────────────────────────────
if submitted:
    pdb_id = pdb_id.strip().upper()
    resname = resname.strip().upper()

    if not pdb_id or not resname:
        st.error("PDB ID and ligand residue name are required.")
        st.stop()

    with st.status(f"Preparing PDB **{pdb_id}** / ligand **{resname}** …",
                  expanded=True) as status:

        # ── Step 1: download ──────────────────────────────────────────
        st.write("📥 **Step 1/5** — Download PDB from RCSB")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_path = Path(pdb_id + ".pdb")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, pdb_path)
            st.write(f"  ✓ Downloaded {url}")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

        # ── Step 2: extract ───────────────────────────────────────────
        st.write("🔍 **Step 2/5** — Extract protein + ligand")

        # Extract ligand HETATM
        lig_count = 0
        with open(pdb_path) as fin, open("ligand_raw.pdb", "w") as fout:
            for line in fin:
                if line.startswith("HETATM") and line[17:20].strip() == resname:
                    fout.write("ATOM  " + line[6:])
                    lig_count += 1

        if lig_count == 0:
            # Try with 4-char residue name
            with open(pdb_path) as fin, open("ligand_raw.pdb", "w") as fout:
                for line in fin:
                    if line.startswith("HETATM") and line[17:21].strip() == resname:
                        fout.write("ATOM  " + line[6:])
                        lig_count += 1

        # Extract protein ATOM records
        prot_count = 0
        with open(pdb_path) as fin, open("receptor_raw.pdb", "w") as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    fout.write(line)
                    prot_count += 1
                elif line.startswith("TER") or line.startswith("END"):
                    fout.write(line)

        st.write(f"  Protein ATOM records: **{prot_count}**")
        st.write(f"  Ligand atoms found: **{lig_count}**")

        if lig_count == 0:
            st.error(f"No HETATM records found for residue '{resname}' in {pdb_id}.pdb")
            st.info("Tip: check the ligand residue code on the RCSB page (e.g. 'H3E', 'TMP', 'STU'). "
                    "Run `grep HETATM <file>.pdb | awk '{print $4}' | sort -u` to list all HETATM residues.")
            pdb_path.unlink(missing_ok=True)
            st.stop()

        # ── Step 3: ligand prep ────────────────────────────────────────
        st.write("🧪 **Step 3/5** — Ligand preparation")

        from rdkit import Chem
        from rdkit.Chem import AllChem
        from openbabel import pybel

        # OpenBabel PDB → MOL
        mol = next(pybel.readfile("pdb", "ligand_raw.pdb"))
        mol.write("mol", "ligand_temp.mol", overwrite=True)

        # RDKit add H + minimize
        rdmol = Chem.MolFromMolFile("ligand_temp.mol", removeHs=True)
        if rdmol is None:
            st.error("RDKit could not parse the ligand from the MOL file.")
            st.stop()

        hmol = Chem.AddHs(rdmol)
        mp = AllChem.MMFFGetMoleculeProperties(hmol)
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(hmol, mp)
            for a in hmol.GetAtoms():
                if a.GetAtomicNum() > 1:
                    ff.MMFFAddPositionConstraint(a.GetIdx(), 0.0, 1.0e4)
            ff.Minimize(maxIts=2000)
        else:
            st.warning("MMFF parameterization failed — skipping minimization.")

        lig_charge = Chem.GetFormalCharge(hmol)
        st.write(f"  Net charge (RDKit): **{lig_charge}**")

        Chem.MolToPDBFile(hmol, "ligand_H.pdb")
        Chem.MolToMolFile(hmol, "ligand.sdf")
        st.write(f"  ✓ ligand.sdf ({hmol.GetNumAtoms()} atoms)")

        # pdb4amber
        subprocess.run(["pdb4amber", "-i", "ligand_H.pdb", "-o", "ligand_h.pdb"],
                       capture_output=True, text=True, timeout=120)
        if not os.path.getsize("ligand_h.pdb"):
            shutil.copy("ligand_H.pdb", "ligand_h.pdb")

        # antechamber
        result = subprocess.run(
            ["antechamber", "-i", "ligand_h.pdb", "-fi", "pdb",
             "-o", "ligand.mol2", "-fo", "mol2",
             "-c", "bcc", "-nc", str(charge),
             "-rn", "LIG", "-at", "gaff2", "-pf", "y"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            st.warning(f"antechamber returned non-zero. stderr:\n```\n{result.stderr[:500]}\n```")

        # parmchk2
        subprocess.run(
            ["parmchk2", "-i", "ligand.mol2", "-f", "mol2",
             "-o", "ligand.frcmod", "-s", "gaff2"],
            capture_output=True, text=True, timeout=120,
        )

        if os.path.getsize("ligand.mol2"):
            st.write(f"  ✓ ligand.mol2 (GAFF2, AM1-BCC)")
        else:
            st.error("antechamber failed to produce ligand.mol2. Check the stderr above.")

        # ── Step 4: protein prep ──────────────────────────────────────
        st.write("🦴 **Step 4/5** — Protein preparation")

        # cpptraj prepareforleap
        import textwrap
        with open("prepareforleap.in", "w") as f:
            f.write(textwrap.dedent(f"""\
                parm receptor_raw.pdb
                loadcrd receptor_raw.pdb name edited
                prepareforleap crdset edited name from-prepareforleap \\
                    pdbout starting1.pdb nowat noh
                go
            """))

        subprocess.run(["cpptraj", "-i", "prepareforleap.in"],
                       capture_output=True, text=True, timeout=120, check=False)

        # pdb4amber
        if os.path.getsize("starting1.pdb"):
            subprocess.run(
                ["pdb4amber", "-i", "starting1.pdb", "-o", "starting_end.pdb", "-a"],
                capture_output=True, text=True, timeout=120, check=False,
            )
        else:
            subprocess.run(
                ["pdb4amber", "-i", "receptor_raw.pdb", "-o", "starting_end.pdb",
                 "--dry", "--most-populous"],
                capture_output=True, text=True, timeout=120, check=False,
            )

        st.write("  ✓ Protein cleaned (cpptraj + pdb4amber)")

        # ── Step 5: add hydrogens ────────────────────────────────────
        st.write("💧 **Step 5/5** — Add protein hydrogens (PDBFixer)")

        import pdbfixer
        from openmm.app import PDBFile

        fixer = pdbfixer.PDBFixer(filename="starting_end.pdb")
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        with open("receptor.pdb", "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

        n_prot = sum(1 for _ in open("receptor.pdb") if _.startswith("ATOM"))
        st.write(f"  ✓ receptor.pdb ({n_prot} atoms)")

        # ── Cleanup intermediates ─────────────────────────────────────
        for f in glob.glob("ANTECHAMBER_*") + glob.glob("ATOMTYPE_*"):
            Path(f).unlink(missing_ok=True)
        for f in ["ligand_raw.pdb", "receptor_raw.pdb", "ligand_temp.mol",
                   "ligand_H.pdb", "ligand_h.pdb", "starting1.pdb",
                   "prepareforleap.in", pdb_id + ".pdb",
                   "ligand_h_renum.txt", "ligand_h_sslink",
                   "starting_end_renum.txt", "starting_end_sslink",
                   "sqm.in", "sqm.out", "sqm.pdb"]:
            Path(f).unlink(missing_ok=True)

        # ── Done ──────────────────────────────────────────────────────
        status.update(label=f"✅ **{pdb_id}** — inputs ready!", state="complete")

    st.divider()
    st.subheader("✅ Results")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if os.path.exists("receptor.pdb"):
            sz = os.path.getsize("receptor.pdb")
            st.success(f"**receptor.pdb**\n\n{sz/1000:.0f} KB\n\n{os.path.getsize('receptor.pdb')/1000:.0f} KB\n{open('receptor.pdb').readline()[:60]}")
    with col_b:
        if os.path.exists("ligand.sdf"):
            sz = os.path.getsize("ligand.sdf")
            st.success(f"**ligand.sdf**\n\n{sz/1000:.1f} KB")
    with col_c:
        if os.path.exists("ligand.mol2"):
            sz = os.path.getsize("ligand.mol2")
            st.success(f"**ligand.mol2**\n\n{sz/1000:.1f} KB")

    st.divider()

    st.subheader("▶️ Next step — run the ABFE simulation")
    st.markdown(
        f"""
The two required files — **`receptor.pdb`** and **`ligand.sdf`** — are
now in the working directory.

Run the ABFE simulation:

```bash
conda activate Dock-MD-FEP
python 02_run_abfe_openfe.py
```

Or run in **Google Colab** / **Jupyter** by copying these two files to your
notebook environment.

---
### What each file is for

| File | Used by | Purpose |
|---|---|---|
| `receptor.pdb` | `02_run_abfe_openfe.py` | Protein with hydrogens (loaded via `ProteinComponent.from_pdb_file()`) |
| `ligand.sdf` | `02_run_abfe_openfe.py` | Ligand with correct element types (loaded via `SmallMoleculeComponent.from_rdkit()`) |
| `ligand.mol2` | AmberTools / tleap | GAFF2-parameterised ligand with AM1-BCC charges |
| `ligand.frcmod` | AmberTools | Force-field modification file |
"""
    )

    # ── Preview ─────────────────────────────────────────────────────────
    with st.expander("👁 Preview files"):
        tab1, tab2 = st.tabs(["receptor.pdb (first 10 lines)", "ligand.sdf"])

        with tab1:
            with open("receptor.pdb") as f:
                lines = [next(f) for _ in range(10)]
            st.code("".join(lines), language="pdb")

        with tab2:
            with open("ligand.sdf") as f:
                content = f.read()
            st.code(content[:500] + "\n...", language="text")

    st.success("You can now close this page. All generated files are in the working directory.")

else:
    # Show usage instructions on first load
    st.info("👆 Fill in the PDB ID and ligand residue name above, then click **Prepare inputs**.")

    with st.expander("📖 How to find the ligand residue name"):
        st.markdown(
            """
            1. Go to [rcsb.org](https://www.rcsb.org) and search your PDB ID
            2. On the structure page, click the **Ligands** tab
            3. The 3-letter code next to your molecule is the residue name

            Or from the command line:
            ```bash
            wget -q -O- https://files.rcsb.org/download/6I5I.pdb | grep HETATM | awk '{print $4}' | sort -u
            ```
            """
        )
