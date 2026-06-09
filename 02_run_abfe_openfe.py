#!/usr/bin/env python3
"""
Absolute Binding Free Energy (ABFE) calculation using OpenFE.

Replaces the legacy Yank-based FEP workflow (Stage 3) in Dock-MD-FEP.

Usage
-----
  conda activate Dock-MD-FEP
  python 02_run_abfe_openfe.py

Expects in the working directory (produced by 01_prepare_inputs.py):
  receptor.pdb  — protein with hydrogens
  ligand.sdf    — small molecule with correct element types
"""

import json, os, sys, time, shutil, logging
from pathlib import Path

from gufe.protocols import execute_DAG
from openff.units import unit as offunit
from rdkit import Chem

from gufe import ChemicalSystem, ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.settings import ThermoSettings, OpenMMSystemGeneratorFFSettings
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AlchemicalSettings, ABFEPreEquilOutputSettings,
    IntegratorSettings, MDSimulationSettings, MultiStateSimulationSettings,
    MultiStateOutputSettings, OpenFFPartialChargeSettings,
    OpenMMEngineSettings, OpenMMSolvationSettings,
)
from openfe.protocols.restraint_utils.settings import BoreschRestraintSettings

# ── paths ──────────────────────────────────────────────────────────
WORK_DIR = Path.cwd()
PROTEIN_PDB = WORK_DIR / "receptor.pdb"
LIGAND_SDF = WORK_DIR / "ligand.sdf"
OUTPUT_DIR = WORK_DIR / "openfe_abfe_output"

# ── logging ──────────────────────────────────────────────────────────
LOG_FILE = WORK_DIR / "abfe_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── simulation parameters ──────────────────────────────────────────
N_REPEATS = 3
TEMPERATURE = 300.0 * offunit.kelvin
PLATFORM = "CUDA"


def check_inputs():
    for f, label in [(PROTEIN_PDB, "receptor.pdb (protein)"),
                     (LIGAND_SDF, "ligand.sdf")]:
        if not f.exists():
            log.error("ERROR: %s not found at %s", label, f)
            sys.exit(1)


def load_protein(pdb_path):
    log.info("  Loading protein from %s ...", pdb_path.name)
    p = ProteinComponent.from_pdb_file(str(pdb_path))
    natoms = p.to_rdkit().GetNumAtoms()
    log.info("  done  (%d atoms)", natoms)
    return p


def load_ligand(sdf_path):
    log.info("  Loading ligand from %s ...", sdf_path.name)

    # Load clean SDF (prepared by 01_prepare_inputs.py with correct element types)
    mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=False))
    if mol is None:
        raise RuntimeError(f"Failed to parse ligand SDF from {sdf_path}")

    lig = SmallMoleculeComponent.from_rdkit(mol)
    log.info("  done  (%s)", lig.smiles)
    return lig


def build_settings():
    """Build AbsoluteBindingSettings, mirroring the original Yank setup."""
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()

    # ── force fields: ff14SB + GAFF2 ───────────────────────────────
    s.forcefield_settings = OpenMMSystemGeneratorFFSettings(
        forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml",
                     "amber/tip3p_HFE_multivalent.xml"],
        small_molecule_forcefield="gaff-2.11",
    )

    # ── solvation: explicit TIP3P with PME ─────────────────────────
    # Note: OpenFE 1.11 does not support GBSA implicit solvent.
    # Explicit solvent is more accurate and well-tested.
    solvent_sol = OpenMMSolvationSettings(solvent_model="tip3p")
    complex_sol = OpenMMSolvationSettings(solvent_model="tip3p")
    s.solvent_solvation_settings = solvent_sol
    s.complex_solvation_settings = complex_sol

    # ── thermodynamics ─────────────────────────────────────────────
    s.thermo_settings = ThermoSettings(temperature=TEMPERATURE,
                                        pressure=1.0 * offunit.bar)

    # ── lambda schedules (use OpenFE defaults, direction is 0→1) ──
    # OpenFE convention: 0 = fully interacting (state A),
    #                    1 = fully decoupled/annihilated (state B)
    # This is the REVERSE of Yank's convention.

    # ── alchemical ─────────────────────────────────────────────────
    s.alchemical_settings = AlchemicalSettings(annihilate_sterics=False)

    # ── restraints ─────────────────────────────────────────────────
    s.restraint_settings = BoreschRestraintSettings()

    # ── engine ─────────────────────────────────────────────────────
    s.engine_settings = OpenMMEngineSettings(compute_platform=PLATFORM)

    # ── integrator ─────────────────────────────────────────────────
    ts = 2.0 * offunit.femtosecond
    s.complex_integrator_settings = IntegratorSettings(timestep=ts)
    s.solvent_integrator_settings = IntegratorSettings(timestep=ts)

    # ── simulation lengths ─────────────────────────────────────────
    s.complex_equil_simulation_settings = MDSimulationSettings(
        equilibration_length=0.1 * offunit.nanosecond,
        production_length=0.1 * offunit.nanosecond,
        equilibration_length_nvt=0.1 * offunit.nanosecond,
    )
    s.solvent_equil_simulation_settings = MDSimulationSettings(
        equilibration_length=0.05 * offunit.nanosecond,
        production_length=0.05 * offunit.nanosecond,
        equilibration_length_nvt=0.05 * offunit.nanosecond,
    )
    n_complex = len(s.complex_lambda_settings.lambda_elec)
    n_solvent = len(s.solvent_lambda_settings.lambda_elec)
    s.complex_simulation_settings = MultiStateSimulationSettings(
        equilibration_length=0.5 * offunit.nanosecond,
        production_length=0.5 * offunit.nanosecond,
        time_per_iteration=2.5 * offunit.picosecond,
        early_termination_target_error=0.20 * offunit.kilocalorie_per_mole,
        real_time_analysis_interval=50.0 * offunit.picosecond,
        n_replicas=n_complex,
    )
    s.solvent_simulation_settings = MultiStateSimulationSettings(
        equilibration_length=0.5 * offunit.nanosecond,
        production_length=0.5 * offunit.nanosecond,
        time_per_iteration=2.5 * offunit.picosecond,
        early_termination_target_error=0.20 * offunit.kilocalorie_per_mole,
        real_time_analysis_interval=50.0 * offunit.picosecond,
        n_replicas=n_solvent,
    )

    # ── output ─────────────────────────────────────────────────────
    s.complex_equil_output_settings = ABFEPreEquilOutputSettings(
        output_indices="all",
    )
    s.solvent_equil_output_settings = ABFEPreEquilOutputSettings(
        output_indices="all",
    )
    s.complex_output_settings = MultiStateOutputSettings(
        output_filename="complex.nc", output_indices="not water",
    )
    s.solvent_output_settings = MultiStateOutputSettings(
        output_filename="solvent.nc", output_indices="not water",
    )

    # ── partial charges ────────────────────────────────────────────
    s.partial_charge_settings = OpenFFPartialChargeSettings(
        partial_charge_method="am1bcc", number_of_conformers=1,
    )

    # ── repeats ────────────────────────────────────────────────────
    s.protocol_repeats = N_REPEATS

    return s


def main():
    log.info("=" * 72)
    log.info("  Dock-MD-FEP → OpenFE   Absolute Binding Free Energy")
    log.info("=" * 72)
    log.info("  Protein : %s", PROTEIN_PDB)
    log.info("  Ligand  : %s", LIGAND_SDF)
    log.info("  Output  : %s", OUTPUT_DIR)
    log.info("  Repeats : %s", N_REPEATS)
    log.info("  Engine  : %s", PLATFORM)

    check_inputs()

    # ── 1 / 4 : Load components ────────────────────────────────────
    log.info("\n── Step 1/4: Loading components ──")
    protein = load_protein(PROTEIN_PDB)
    ligand = load_ligand(LIGAND_SDF)
    solvent = SolventComponent()  # 0.15 M NaCl, neutralized

    # ── 2 / 4 : Build settings ──────────────────────────────────────
    log.info("\n── Step 2/4: Configuring ABFE protocol settings ──")
    settings = build_settings()
    n_complex = len(settings.complex_lambda_settings.lambda_elec)
    n_solvent = len(settings.solvent_lambda_settings.lambda_elec)
    log.info("  Complex λ windows : %s", n_complex)
    log.info("  Solvent λ windows : %s", n_solvent)

    # ── 3 / 4 : Create protocol & DAG ───────────────────────────────
    log.info("\n── Step 3/4: Creating ABFE transformation ──")
    stateA = ChemicalSystem(
        {"protein": protein, "ligand": ligand, "solvent": solvent},
        name="complex",
    )
    stateB = ChemicalSystem(
        {"protein": protein, "solvent": solvent},
        name="protein_only",
    )

    protocol = openmm_afe.AbsoluteBindingProtocol(settings=settings)
    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None,
                          name="Dock-MD-FEP_ABFE")

    n_units = len(dag.protocol_units)
    log.info("  Protocol units : %s", n_units)

    # ── 4 / 4 : Run ─────────────────────────────────────────────────
    log.info("\n── Step 4/4: Running simulation ──")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "shared").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "scratch").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dag_result = execute_DAG(
        dag,
        shared_basedir=OUTPUT_DIR / "shared",
        scratch_basedir=OUTPUT_DIR / "scratch",
    )
    elapsed = time.time() - t0

    # ── Results ─────────────────────────────────────────────────────
    log.info("\n── Results ──")
    log.info("  Wall time : %ds (%1.f min)", elapsed, elapsed / 60.0)

    # dag_result is a ProtocolDAGResult.
    # Use protocol.gather() to aggregate into a AbsoluteBindingProtocolResult.
    try:
        prot_result = protocol.gather([dag_result])
    except Exception as e:
        log.warning("  protocol.gather() failed: %s", e)
        prot_result = None

    summary = {}
    if prot_result is not None:
        try:
            dg_estimate = prot_result.get_estimate()
            dg_uncertainty = prot_result.get_uncertainty()

            kT = dg_estimate.magnitude
            kT_err = dg_uncertainty.magnitude

            # kT at 300 K = 0.596 kcal/mol
            kT_per_kcal = 0.596
            dg_kcal = kT * kT_per_kcal
            dg_err_kcal = kT_err * kT_per_kcal

            # NOTE: dg_estimate from get_estimate() is already the fully
            # corrected ΔG°, including the Boresch standard state correction.
            # The correction is applied internally by OpenFE (see
            # AbsoluteBindingProtocolResult.get_estimate()).

            log.info("\n  ΔG° (kT)       = %.3f ± %.3f kT", kT, kT_err)
            log.info("  ΔG° (kcal/mol) = %.3f ± %.3f kcal/mol", dg_kcal, dg_err_kcal)

            summary = {
                "dg_kT": kT,
                "dg_kcal_per_mol": dg_kcal,
                "dg_error_kT": kT_err,
                "dg_error_kcal_per_mol": dg_err_kcal,
                "temperature_K": 300.0,
                "platform": PLATFORM,
                "n_repeats": N_REPEATS,
            }
        except Exception as e:
            log.error("  Error extracting results: %s", e)
            summary = {"error": str(e)}
    else:
        log.warning("  WARNING: no protocol_result found in DAG result.")
        log.warning("  DAG result type: %s", type(dag_result).__name__)
        log.warning("  Attributes: %s", [a for a in dir(dag_result) if not a.startswith('_')])

    # Save summary
    out = WORK_DIR / "abfe_results.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("\n  Results → %s", out)

    # Clean scratch
    sp = OUTPUT_DIR / "scratch"
    if sp.exists():
        shutil.rmtree(sp, ignore_errors=True)

    log.info("\n✓ Done")


if __name__ == "__main__":
    main()
