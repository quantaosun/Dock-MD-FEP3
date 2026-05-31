# ABFE with OpenFE

Absolute Binding Free Energy calculation from any PDB structure.

```
         ┌──────────────────────────────────────────────────┐
         │         WHAT IS ABFE?                            │
         │                                                  │
         │  ΔG°(binding) = ΔG(complex) - ΔG(solvent)       │
         │       ↘              ↙                            │
         │  Alchemically decouple ligand in both            │
         │  environments → MBAR → ΔG°                       │
         └──────────────────────────────────────────────────┘
```

## Pipeline

```
  PDB ID                          ┌─────────────────────┐
     │                            │ 01_prepare_inputs.py│
     ├── protein ATOM records ───→│ OpenBabel → RDKit → │
     │                            │ pdb4amber → antech  │
     ├── ligand HETATM ──────────→│ amber(GAFF2/AM1-BCC)│
     │                            │ → PDBFixer (add H)  │
     │                            └─────────┬───────────┘
     │                                      │
     │                            ┌─────────▼───────────┐
     │                            │ receptor.pdb        │
     │                            │ ligand.sdf           │
     │                            └─────────┬───────────┘
     │                                      │
     │                            ┌─────────▼───────────┐
     │                            │ 02_run_abfe_openfe.py│
     ├── Solvent leg ────────────→│ solvent/14 λ windows │
     ├── Complex leg ────────────→│ complex/30 λ windows │
     │                            │ → MBAR → ΔG°        │
     │                            └─────────┬───────────┘
     │                                      │
     │                            ┌─────────▼───────────┐
     │                            │ abfe_results.json   │
     └────────────────────────────│ ΔG° in kcal/mol     │
                                  └─────────────────────┘
```

## Quick start

```bash
conda env create -f environment.yml
conda activate Dock-MD-FEP

# Option A — CLI
python 01_prepare_inputs.py --pdb 6I5I --resname H3E
python 02_run_abfe_openfe.py

# Option B — Web UI
streamlit run app.py
```

## How ABFE works (thermodynamic cycle)

```
                 ┌──────────────────────────────┐
                 │  ΔG°(binding) = ΔG*_complex  │
                 │                - ΔG*_solvent │
                 │                + ΔG°_std_corr│
                 └──────────────┬───────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
  │ Protein +    │     │ Protein only │      │  Ligand in   │
  │ Ligand in    │ ←── │ (reference)  │      │  water       │
  │ water        │     │              │      │              │
  └──────┬───────┘     └──────────────┘      └──────┬───────┘
         │  ΔG*_complex                             │  ΔG*_solvent
         │  (alchemical decoupling                  │  (alchemical
         │   in the binding site)                   │   decoupling
         ▼                                          ▼
  ┌──────────────┐                           ┌──────────────┐
  │ Protein +    │                           │  Ligand in   │
  │ decoupled    │                           │  decoupled   │
  │ ligand (gas) │                           │  (gas phase) │
  └──────────────┘                           └──────────────┘
         │
         └── ΔG°_std_corr = RT ln(C°V°) ≈ 2.38 kcal/mol

  OpenFE  λ=0: fully interacting (bound / solvated)
          λ=1: fully decoupled (no nonbonded interactions)
```

## Input preparation

For **any** protein–ligand complex PDB:

```bash
python 01_prepare_inputs.py --pdb <PDB_ID> --resname <LIGAND> [--charge <N>]
```

| Argument | What it is |
|---|---|
| `--pdb` | 4-character PDB ID (e.g. `6I5I`) or a local file stem |
| `--resname` | The HETATM residue code of your ligand (e.g. `H3E`, `TMP`) |
| `--charge` | Formal charge of the ligand (default `0`) |

**Find the residue name:**
```bash
wget -q -O- https://files.rcsb.org/download/6I5I.pdb | grep HETATM | awk '{print $4}' | sort -u
```

**Examples:**
```bash
# CLK1 kinase + H3E inhibitor (this repo's tested example)
python 01_prepare_inputs.py --pdb 6I5I --resname H3E

# T4 lysozyme + benzene (PDB 181L)
python 01_prepare_inputs.py --pdb 181L --resname TMP
```

## Custom PDB (not from RCSB)

```bash
cp my_complex.pdb my.pdb
python 01_prepare_inputs.py --pdb my --resname XYZ
```

Input files (`receptor.pdb` + `ligand.sdf`) are written to the **current directory**, ready for step 2.

## Running the simulation

```bash
python 02_run_abfe_openfe.py
```

- Solvent leg: 14 λ windows (ligand → vacuum in water)
- Complex leg: 30 λ windows (ligand → vacuum in binding site)
- Results via MBAR (`protocol.gather()`) → `abfe_results.json`

**Customise** at the top of the script:
- `PLATFORM` — `"CUDA"` or `"CPU"`
- `N_REPEATS` — set ≥3 for error bars
- Simulation lengths — increase for convergence

## Output

| File | Contents |
|---|---|
| `abfe_results.json` | ΔG° in kT and kcal/mol |
| `abfe_run.log` | Full simulation log |
| `openfe_abfe_output/` | Checkpoints, trajectories, restart data |

## Example result

Benzene → 3HTB on Blackwell GPU (~141 min):

```json
{
  "dg_kT": -5.951,
  "dg_kcal_per_mol": -3.547,
  "dg_0_kcal_per_mol": -3.547
}
```

## Environment

Install via `environment.yml`. Requires OpenFE ≥1.11, AmberTools, OpenMM.

> **Blackwell GPUs (CC 12.0):** OpenMM must be built from source — pre-built conda wheels lack PTX. See § *Building OpenMM from source* in the full guide or `environment.yml` for instructions.

## License

MIT — see LICENSE.
