#!/usr/bin/env python3
"""
Analyze paired full/reduced PINN-CAC history files and generate heatmaps for
RESET and non-RESET experiments.

Expected filename forms
-----------------------
pinn-history-2d-alloy-neumann-full-IL0-BL1-TRIAL1.json
pinn-history-2d-alloy-neumann-reduced-IL0-BL1-TRIAL1.json

pinn-history-2d-alloy-neumann-full-IL0-BL1-RESET-TRIAL1.json
pinn-history-2d-alloy-neumann-reduced-IL0-BL1-RESET-TRIAL1.json

The script:
1. inventories all matching files;
2. reports missing, duplicate, suspiciously small, and corrupt files;
3. pairs each reduced run with its corresponding full-atom run;
4. extracts final-state errors and representative-atom counts;
5. writes trial-level and aggregated CSV files;
6. generates separate heatmaps for RESET and non-RESET runs.

Only one full/reduced pair is loaded at a time, which is important when each
history file is tens of megabytes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Load LAMMPS trajectory
# ---------------------------------------------------------------------------
from shared_lammps_2D import read_lammps_trajectory_xy

LAMMPS_TRAJ_FILE = "./lammps_two_material/two_material.dump"

lammps_timesteps, lammps_frames = read_lammps_trajectory_xy(
    LAMMPS_TRAJ_FILE
)

lammps_final_pos = np.asarray(
    lammps_frames[-1],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Final-energy outlier rejection controls
# ---------------------------------------------------------------------------

# Outliers are detected independently inside each:
# (interface_layers, boundary_layers, reset) group.
OUTLIER_METRIC = "relative_true_energy_error"

# Conservative robust-outlier threshold.
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5

# Do not perform group-based outlier detection with fewer than this many
# otherwise valid trials.
OUTLIER_MIN_GROUP_SIZE = 4

# Optional hard cutoff. For example, use 0.10 to reject any trial whose
# final relative energy error is greater than 10%.
# Set to None to disable the hard cutoff.
OUTLIER_ABSOLUTE_THRESHOLD = 0.1

# Used only when the median absolute deviation is zero.
OUTLIER_IQR_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# User-editable defaults
# ---------------------------------------------------------------------------


DEFAULT_HISTORY_DIR = Path("./pinn_history")
DEFAULT_OUTPUT_DIR = Path("./pinn_analysis")

NX = 49
NY = 50
LATTICE_SPACING = 2 ** (1 / 6)

EXPECTED_INTERFACE_LAYERS = (0, 1, 2, 3)
EXPECTED_BOUNDARY_LAYERS = (1, 2, 3)
EXPECTED_TRIALS = tuple(range(1, 11))
EXPECTED_RESET_VALUES = (False, True)

# A valid 30 MB history is much larger than this. This catches files such as
# "[]" or truncated outputs before attempting to parse them.
MIN_VALID_FILE_SIZE_BYTES = 1000

# Set to an integer such as 19900 when all valid histories must end at that
# exact logged epoch. Leave as None to accept any finite final epoch.
EXPECTED_FINAL_EPOCH: int | None = None

# Optional physical failure criteria. Leave as None to disable.
MAX_DISPLACEMENT_ALLOWED: float | None = None
MIN_DETF_ALLOWED: float | None = None

EPS = 1.0e-14


FILENAME_RE = re.compile(
    r"^pinn-history-2d-alloy-neumann-"
    r"(?P<kind>full|reduced)-"
    r"IL(?P<interface_layers>\d+)-"
    r"BL(?P<boundary_layers>\d+)-"
    r"(?P<reset>RESET-)?"
    r"TRIAL(?P<trial>\d+)\.json$"
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def make_triangular_lattice(nx: int, ny: int, spacing: float) -> np.ndarray:
    """
    Generate the same row-major triangular lattice used by the simulations.

    If your shared_2D.make_triangular_lattice uses a different ordering or
    origin, replace this function with that exact implementation.
    """
    positions = []

    dy = np.sqrt(3.0) * spacing / 2.0

    for j in range(ny):
        x_shift = 0.5 * spacing * (j % 2)

        for i in range(nx):
            positions.append(
                [
                    i * spacing + x_shift,
                    j * dy,
                ]
            )

    return np.asarray(positions, dtype=np.float64)


# ---------------------------------------------------------------------------
# Filename parsing and inventory
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> dict[str, Any] | None:
    match = FILENAME_RE.match(path.name)

    if match is None:
        return None

    fields = match.groupdict()

    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "kind": fields["kind"],
        "interface_layers": int(fields["interface_layers"]),
        "boundary_layers": int(fields["boundary_layers"]),
        "reset": fields["reset"] is not None,
        "trial": int(fields["trial"]),
        "file_size_bytes": path.stat().st_size,
    }


def build_inventory(history_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(history_dir.glob("*.json")):
        parsed = parse_filename(path)

        if parsed is not None:
            rows.append(parsed)

    columns = [
        "path",
        "filename",
        "kind",
        "interface_layers",
        "boundary_layers",
        "reset",
        "trial",
        "file_size_bytes",
    ]

    return pd.DataFrame(rows, columns=columns)


def expected_run_keys() -> pd.DataFrame:
    rows = []

    for il in EXPECTED_INTERFACE_LAYERS:
        for bl in EXPECTED_BOUNDARY_LAYERS:
            for reset in EXPECTED_RESET_VALUES:
                for trial in EXPECTED_TRIALS:
                    rows.append(
                        {
                            "interface_layers": il,
                            "boundary_layers": bl,
                            "reset": reset,
                            "trial": trial,
                        }
                    )

    return pd.DataFrame(rows)


def validate_inventory(inventory: pd.DataFrame) -> None:
    if inventory.empty:
        raise RuntimeError(
            "No matching history files were found. Check HISTORY_DIR and the "
            "filename pattern."
        )

    key_cols = [
        "kind",
        "interface_layers",
        "boundary_layers",
        "reset",
        "trial",
    ]

    duplicate_mask = inventory.duplicated(key_cols, keep=False)
    duplicates = inventory.loc[duplicate_mask].sort_values(key_cols)

    if not duplicates.empty:
        print("\nWARNING: duplicate experiment files:")
        print(duplicates[key_cols + ["path"]].to_string(index=False))

    suspicious = inventory[
        inventory["file_size_bytes"] < MIN_VALID_FILE_SIZE_BYTES
    ]

    if not suspicious.empty:
        print("\nWARNING: suspiciously small files:")
        print(
            suspicious[
                [
                    "kind",
                    "interface_layers",
                    "boundary_layers",
                    "reset",
                    "trial",
                    "file_size_bytes",
                    "path",
                ]
            ].to_string(index=False)
        )

    expected = expected_run_keys()

    for kind in ("full", "reduced"):
        observed = inventory[inventory["kind"] == kind][
            ["interface_layers", "boundary_layers", "reset", "trial"]
        ].drop_duplicates()

        missing = expected.merge(
            observed,
            on=["interface_layers", "boundary_layers", "reset", "trial"],
            how="left",
            indicator=True,
        )

        missing = missing[missing["_merge"] == "left_only"].drop(
            columns="_merge"
        )

        if not missing.empty:
            print(f"\nWARNING: missing {kind} histories:")
            print(missing.to_string(index=False))


# ---------------------------------------------------------------------------
# History loading and final-state validation
# ---------------------------------------------------------------------------

def load_history(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    try:
        if path.stat().st_size < MIN_VALID_FILE_SIZE_BYTES:
            return None, "suspiciously_small_file"

        with path.open("r") as stream:
            history = json.load(stream)

    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"

    except OSError as exc:
        return None, f"io_error: {exc}"

    if not isinstance(history, list):
        return None, "history_root_is_not_a_list"

    if len(history) == 0:
        return None, "empty_history"

    if not isinstance(history[-1], dict):
        return None, "final_frame_is_not_an_object"

    return history, ""


def validate_final_frame(
    final: dict[str, Any],
    atom_pos: np.ndarray,
    require_rep_indices: bool,
) -> tuple[bool, str]:
    required = {"epoch", "pos", "true_energy", "loss_energy", "detF"}

    if require_rep_indices:
        required.add("rep_indices")

    missing = sorted(required - set(final))

    if missing:
        return False, f"missing_final_keys:{','.join(missing)}"

    try:
        pos = np.asarray(final["pos"], dtype=np.float64)
        detf = np.asarray(final["detF"], dtype=np.float64)
        epoch = float(final["epoch"])
        true_energy = float(final["true_energy"])
        loss_energy = float(final["loss_energy"])

    except (TypeError, ValueError) as exc:
        return False, f"invalid_final_value:{exc}"

    if pos.shape != atom_pos.shape:
        return (
            False,
            f"position_shape_mismatch:{pos.shape}!={atom_pos.shape}",
        )

    if detf.shape != (len(atom_pos),):
        return (
            False,
            f"detF_shape_mismatch:{detf.shape}!={(len(atom_pos),)}",
        )

    if not np.all(np.isfinite(pos)):
        return False, "nonfinite_positions"

    if not np.all(np.isfinite(detf)):
        return False, "nonfinite_detF"

    if not np.isfinite(epoch):
        return False, "nonfinite_epoch"

    if not np.isfinite(true_energy):
        return False, "nonfinite_true_energy"

    if not np.isfinite(loss_energy):
        return False, "nonfinite_loss_energy"

    if EXPECTED_FINAL_EPOCH is not None:
        if int(epoch) != EXPECTED_FINAL_EPOCH:
            return False, f"unexpected_final_epoch:{int(epoch)}"

    max_displacement = np.linalg.norm(pos - atom_pos, axis=1).max()

    if MAX_DISPLACEMENT_ALLOWED is not None:
        if max_displacement > MAX_DISPLACEMENT_ALLOWED:
            return False, f"displacement_runaway:{max_displacement:.6e}"

    if MIN_DETF_ALLOWED is not None:
        if np.min(detf) < MIN_DETF_ALLOWED:
            return False, f"detF_below_limit:{np.min(detf):.6e}"

    return True, ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def relative_l2_error(
    reference: np.ndarray,
    prediction: np.ndarray,
    eps: float = EPS,
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    numerator = np.linalg.norm(prediction - reference)
    denominator = np.linalg.norm(reference)

    return float(numerator / max(denominator, eps))


def relative_scalar_error(
    reference: float,
    prediction: float,
    eps: float = EPS,
) -> float:
    return float(
        abs(prediction - reference)
        / max(abs(reference), eps)
    )


def count_rep_count_changes(history: list[dict[str, Any]]) -> int:
    counts = []

    for frame in history:
        rep_indices = frame.get("rep_indices", [])
        counts.append(len(rep_indices))

    if len(counts) < 2:
        return 0

    return int(np.count_nonzero(np.diff(counts)))


def summarize_pair(
    metadata: dict[str, Any],
    full_path: Path,
    reduced_path: Path,
    atom_pos: np.ndarray,
    lammps_final_pos: np.ndarray,
) -> dict[str, Any]:
    row = {
        **metadata,
        "full_path": str(full_path),
        "reduced_path": str(reduced_path),
        "failed": True,
        "failure_reason": "",
    }

    full_history, full_error = load_history(full_path)

    if full_history is None:
        row["failure_reason"] = f"full:{full_error}"
        return row

    reduced_history, reduced_error = load_history(reduced_path)

    if reduced_history is None:
        row["failure_reason"] = f"reduced:{reduced_error}"
        return row

    full_final = full_history[-1]
    reduced_final = reduced_history[-1]

    full_valid, full_reason = validate_final_frame(
        full_final,
        atom_pos=atom_pos,
        require_rep_indices=False,
    )

    if not full_valid:
        row["failure_reason"] = f"full:{full_reason}"
        return row

    reduced_valid, reduced_reason = validate_final_frame(
        reduced_final,
        atom_pos=atom_pos,
        require_rep_indices=True,
    )

    if not reduced_valid:
        row["failure_reason"] = f"reduced:{reduced_reason}"
        return row

    pos_full = np.asarray(full_final["pos"], dtype=np.float64)
    pos_reduced = np.asarray(reduced_final["pos"], dtype=np.float64)

    u_lammps = lammps_final_pos - atom_pos
    u_full = pos_full - atom_pos
    u_reduced = pos_reduced - atom_pos

    detf_full = np.asarray(full_final["detF"], dtype=np.float64)
    detf_reduced = np.asarray(reduced_final["detF"], dtype=np.float64)

    q_full = np.asarray(
        full_final.get("q_full", np.full(len(atom_pos), np.nan)),
        dtype=np.float64,
    )
    q_reduced = np.asarray(
        reduced_final.get("q_full", np.full(len(atom_pos), np.nan)),
        dtype=np.float64,
    )

    final_rep_indices = np.asarray(
        reduced_final.get("rep_indices", []),
        dtype=int,
    )

    initial_rep_indices = np.asarray(
        reduced_history[0].get("rep_indices", []),
        dtype=int,
    )

    full_true_energy = float(full_final["true_energy"])
    reduced_true_energy = float(reduced_final["true_energy"])
    reduced_loss_energy = float(reduced_final["loss_energy"])

    row.update(
        {
            "failed": False,
            "failure_reason": "",
            "n_full_frames": len(full_history),
            "n_reduced_frames": len(reduced_history),
            "full_final_epoch": int(full_final["epoch"]),
            "reduced_final_epoch": int(reduced_final["epoch"]),

            # Primary accuracy metrics
            "relative_displacement_error": relative_l2_error(
                u_lammps,
                u_reduced,
            ),
            # Explicitly named equivalent
            "relative_reduced_vs_lammps_error": relative_l2_error(
                u_lammps,
                u_reduced,
            ),
            # Full PINN baseline accuracy
            "relative_full_vs_lammps_error": relative_l2_error(
                u_lammps,
                u_full,
            ),
            # Additional error introduced by model reduction
            "relative_reduced_vs_full_error": relative_l2_error(
                u_full,
                u_reduced,
            ),
            
            "relative_detF_error": relative_l2_error(
                detf_full,
                detf_reduced,
            ),
            "relative_true_energy_error": relative_scalar_error(
                full_true_energy,
                reduced_true_energy,
            ),

            # Quadrature inconsistency inside the reduced simulation
            "relative_reduced_loss_true_error": relative_scalar_error(
                reduced_true_energy,
                reduced_loss_energy,
            ),

            # Additional diagnostics
            "mean_absolute_position_error": float(
                np.mean(np.linalg.norm(pos_reduced - pos_full, axis=1))
            ),
            "max_absolute_position_error": float(
                np.max(np.linalg.norm(pos_reduced - pos_full, axis=1))
            ),
            "max_full_displacement": float(
                np.max(np.linalg.norm(u_full, axis=1))
            ),
            "max_reduced_displacement": float(
                np.max(np.linalg.norm(u_reduced, axis=1))
            ),
            "min_full_detF": float(np.min(detf_full)),
            "min_reduced_detF": float(np.min(detf_reduced)),
            "full_true_energy": full_true_energy,
            "reduced_true_energy": reduced_true_energy,
            "reduced_loss_energy": reduced_loss_energy,

            # Representative atom statistics
            "initial_rep_atoms": int(len(initial_rep_indices)),
            "final_rep_atoms": int(len(final_rep_indices)),
            "rep_atoms_spawned": int(
                len(final_rep_indices) - len(initial_rep_indices)
            ),
            "rep_count_change_events": count_rep_count_changes(
                reduced_history
            ),
        }
    )

    if (
        q_full.shape == (len(atom_pos),)
        and q_reduced.shape == (len(atom_pos),)
        and np.all(np.isfinite(q_full))
        and np.all(np.isfinite(q_reduced))
    ):
        row["relative_q_error"] = relative_l2_error(
            q_full,
            q_reduced,
        )
    else:
        row["relative_q_error"] = np.nan

    return row


# ---------------------------------------------------------------------------
# Pairing and aggregation
# ---------------------------------------------------------------------------

def pair_histories(inventory: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "interface_layers",
        "boundary_layers",
        "reset",
        "trial",
    ]

    # Drop exact-key duplicates only after warning. Keeping the first makes the
    # behavior deterministic while still allowing the analysis to proceed.
    deduplicated = inventory.drop_duplicates(
        ["kind", *key_cols],
        keep="first",
    )

    full = deduplicated[deduplicated["kind"] == "full"][
        key_cols + ["path", "file_size_bytes"]
    ].rename(
        columns={
            "path": "full_path",
            "file_size_bytes": "full_file_size_bytes",
        }
    )

    reduced = deduplicated[deduplicated["kind"] == "reduced"][
        key_cols + ["path", "file_size_bytes"]
    ].rename(
        columns={
            "path": "reduced_path",
            "file_size_bytes": "reduced_file_size_bytes",
        }
    )

    return full.merge(reduced, on=key_cols, how="outer")


def analyze_all_pairs(
    paired: pd.DataFrame,
    atom_pos: np.ndarray,
    lammps_final_pos:np.ndarray,
) -> pd.DataFrame:
    rows = []

    for index, pair in paired.sort_values(
        ["reset", "interface_layers", "boundary_layers", "trial"]
    ).iterrows():
        metadata = {
            "interface_layers": int(pair["interface_layers"]),
            "boundary_layers": int(pair["boundary_layers"]),
            "reset": bool(pair["reset"]),
            "trial": int(pair["trial"]),
        }

        full_missing = pd.isna(pair.get("full_path"))
        reduced_missing = pd.isna(pair.get("reduced_path"))

        if full_missing or reduced_missing:
            missing = []

            if full_missing:
                missing.append("full")

            if reduced_missing:
                missing.append("reduced")

            rows.append(
                {
                    **metadata,
                    "full_path": pair.get("full_path", ""),
                    "reduced_path": pair.get("reduced_path", ""),
                    "failed": True,
                    "failure_reason": "missing_" + "_and_".join(missing),
                }
            )
            continue

        print(
            "Analyzing "
            f"IL={metadata['interface_layers']} "
            f"BL={metadata['boundary_layers']} "
            f"reset={metadata['reset']} "
            f"trial={metadata['trial']}"
        )

        rows.append(
            summarize_pair(
                metadata=metadata,
                full_path=Path(pair["full_path"]),
                reduced_path=Path(pair["reduced_path"]),
                atom_pos=atom_pos,
                lammps_final_pos=lammps_final_pos,
            )
        )

    return pd.DataFrame(rows)

def mark_final_energy_outliers(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Mark trials with abnormally large final relative true-energy errors.

    Outliers are detected separately for every combination of:

        interface_layers
        boundary_layers
        reset

    Only trials that have already passed the file and final-frame validation
    participate in outlier detection.

    The primary method is a one-sided modified z-score based on the median
    absolute deviation (MAD). Since relative energy error is nonnegative,
    only unusually large values are rejected.

    If MAD is zero, the function falls back to an upper IQR criterion.
    """
    result = summary.copy()

    result["energy_outlier"] = False
    result["energy_outlier_method"] = ""
    result["energy_outlier_score"] = np.nan
    result["energy_outlier_upper_limit"] = np.nan

    group_cols = [
        "interface_layers",
        "boundary_layers",
        "reset",
    ]

    valid_mask = (
        (~result["failed"])
        & result[OUTLIER_METRIC].notna()
        & np.isfinite(result[OUTLIER_METRIC])
    )

    valid = result.loc[valid_mask]

    for group_key, group in valid.groupby(group_cols, dropna=False):
        indices = group.index
        values = group[OUTLIER_METRIC].astype(float)

        # ---------------------------------------------------------------
        # Optional hard absolute cutoff
        # ---------------------------------------------------------------
        if OUTLIER_ABSOLUTE_THRESHOLD is not None:
            hard_mask = values > OUTLIER_ABSOLUTE_THRESHOLD
            hard_indices = indices[hard_mask]

            result.loc[hard_indices, "energy_outlier"] = True
            result.loc[
                hard_indices,
                "energy_outlier_method",
            ] = "absolute_threshold"

            result.loc[
                hard_indices,
                "energy_outlier_upper_limit",
            ] = OUTLIER_ABSOLUTE_THRESHOLD

        # A statistical test is unreliable for very small groups.
        if len(values) < OUTLIER_MIN_GROUP_SIZE:
            continue

        median = float(np.median(values))
        absolute_deviation = np.abs(values - median)
        mad = float(np.median(absolute_deviation))

        # ---------------------------------------------------------------
        # Primary test: one-sided modified z-score
        # ---------------------------------------------------------------
        if mad > 0.0:
            modified_z = (
                0.6744897501960817
                * (values - median)
                / mad
            )

            upper_limit = (
                median
                + OUTLIER_MODIFIED_Z_THRESHOLD
                * mad
                / 0.6744897501960817
            )

            result.loc[
                indices,
                "energy_outlier_score",
            ] = modified_z.to_numpy()

            result.loc[
                indices,
                "energy_outlier_upper_limit",
            ] = upper_limit

            statistical_mask = (
                modified_z > OUTLIER_MODIFIED_Z_THRESHOLD
            )

            statistical_indices = indices[statistical_mask]

            result.loc[
                statistical_indices,
                "energy_outlier",
            ] = True

            # Preserve the absolute-threshold label when both tests reject
            # the same trial.
            unlabeled = statistical_indices[
                result.loc[
                    statistical_indices,
                    "energy_outlier_method",
                ] == ""
            ]

            result.loc[
                unlabeled,
                "energy_outlier_method",
            ] = "modified_z"

        # ---------------------------------------------------------------
        # Fallback test when MAD == 0
        # ---------------------------------------------------------------
        else:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1

            if iqr <= 0.0:
                continue

            upper_limit = (
                q3
                + OUTLIER_IQR_MULTIPLIER * iqr
            )

            result.loc[
                indices,
                "energy_outlier_upper_limit",
            ] = upper_limit

            statistical_mask = values > upper_limit
            statistical_indices = indices[statistical_mask]

            result.loc[
                statistical_indices,
                "energy_outlier",
            ] = True

            unlabeled = statistical_indices[
                result.loc[
                    statistical_indices,
                    "energy_outlier_method",
                ] == ""
            ]

            result.loc[
                unlabeled,
                "energy_outlier_method",
            ] = "iqr"

        il, bl, reset = group_key

        rejected = result.loc[
            indices,
            "energy_outlier",
        ].sum()

        print(
            "Outlier check: "
            f"IL={il}, BL={bl}, reset={reset}, "
            f"valid_trials={len(values)}, rejected={rejected}"
        )

    result["excluded_from_statistics"] = (
        result["failed"]
        | result["energy_outlier"]
    )

    result["exclusion_reason"] = ""

    result.loc[
        result["failed"],
        "exclusion_reason",
    ] = result.loc[
        result["failed"],
        "failure_reason",
    ]

    result.loc[
        (~result["failed"]) & result["energy_outlier"],
        "exclusion_reason",
    ] = "final_energy_outlier"

    return result


def quantile_90(values: pd.Series) -> float:
    values = values.dropna()

    if len(values) == 0:
        return np.nan

    return float(values.quantile(0.90))


def aggregate_trials_old(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "interface_layers",
        "boundary_layers",
        "reset",
    ]

    grouped = summary.groupby(group_cols, dropna=False)

    aggregate = grouped.agg(
        expected_or_found_trials=("trial", "count"),
        failed_trials=("failed", "sum"),
        failure_rate=("failed", "mean"),
    ).reset_index()

    successful = summary[~summary["failed"]].copy()

    if successful.empty:
        return aggregate

    accuracy = (
        successful.groupby(group_cols)
        .agg(
            successful_trials=("trial", "count"),

            median_displacement_error=(
                "relative_displacement_error",
                "median",
            ),
            q90_displacement_error=(
                "relative_displacement_error",
                quantile_90,
            ),
            median_detF_error=(
                "relative_detF_error",
                "median",
            ),
            q90_detF_error=(
                "relative_detF_error",
                quantile_90,
            ),
            median_true_energy_error=(
                "relative_true_energy_error",
                "median",
            ),
            q90_true_energy_error=(
                "relative_true_energy_error",
                quantile_90,
            ),
            median_quadrature_error=(
                "relative_reduced_loss_true_error",
                "median",
            ),
            median_final_rep_atoms=(
                "final_rep_atoms",
                "median",
            ),
            q90_final_rep_atoms=(
                "final_rep_atoms",
                quantile_90,
            ),
            median_rep_atoms_spawned=(
                "rep_atoms_spawned",
                "median",
            ),
        )
        .reset_index()
    )

    return aggregate.merge(
        accuracy,
        on=group_cols,
        how="left",
    )


def aggregate_trials(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trials after excluding:

    1. invalid or failed trials;
    2. trials marked as final-energy outliers.

    A parameter combination is allowed to have fewer than ten accepted trials.
    """
    group_cols = [
        "interface_layers",
        "boundary_layers",
        "reset",
    ]

    grouped = summary.groupby(group_cols, dropna=False)

    aggregate = grouped.agg(
        found_trials=("trial", "count"),
        failed_trials=("failed", "sum"),
        energy_outlier_trials=("energy_outlier", "sum"),
        excluded_trials=("excluded_from_statistics", "sum"),
    ).reset_index()

    aggregate["raw_failure_rate"] = (
        aggregate["failed_trials"]
        / aggregate["found_trials"]
    )

    accepted = summary[
        ~summary["excluded_from_statistics"]
    ].copy()

    accepted_counts = (
        accepted.groupby(group_cols)
        .size()
        .rename("accepted_trials")
        .reset_index()
    )

    aggregate = aggregate.merge(
        accepted_counts,
        on=group_cols,
        how="left",
    )

    aggregate["accepted_trials"] = (
        aggregate["accepted_trials"]
        .fillna(0)
        .astype(int)
    )

    aggregate["acceptance_fraction"] = (
        aggregate["accepted_trials"]
        / aggregate["found_trials"]
    )

    if accepted.empty:
        return aggregate

    accuracy = (
        accepted.groupby(group_cols)
        .agg(
            median_reduced_vs_lammps_error=(
                "relative_reduced_vs_lammps_error",
                "median",
            ),
            q90_reduced_vs_lammps_error=(
                "relative_reduced_vs_lammps_error",
                quantile_90,
            ),
            median_full_vs_lammps_error=(
                "relative_full_vs_lammps_error",
                "median",
            ),
            median_reduced_vs_full_error=(
                "relative_reduced_vs_full_error",
                "median",
            ),
            median_displacement_error=(
                "relative_displacement_error",
                "median",
            ),
            q90_displacement_error=(
                "relative_displacement_error",
                quantile_90,
            ),
            median_detF_error=(
                "relative_detF_error",
                "median",
            ),
            q90_detF_error=(
                "relative_detF_error",
                quantile_90,
            ),
            median_true_energy_error=(
                "relative_true_energy_error",
                "median",
            ),
            q90_true_energy_error=(
                "relative_true_energy_error",
                quantile_90,
            ),
            median_quadrature_error=(
                "relative_reduced_loss_true_error",
                "median",
            ),
            median_final_rep_atoms=(
                "final_rep_atoms",
                "median",
            ),
            q90_final_rep_atoms=(
                "final_rep_atoms",
                quantile_90,
            ),
            median_rep_atoms_spawned=(
                "rep_atoms_spawned",
                "median",
            ),
        )
        .reset_index()
    )

    return aggregate.merge(
        accuracy,
        on=group_cols,
        how="left",
    )

# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

HEATMAP_SPECS = {
    "median_displacement_error": {
        "title": "Median relative displacement error",
        "label": "Relative error",
        "percent": True,
        "log": True,
    },
    "q90_displacement_error": {
        "title": "90th-percentile relative displacement error",
        "label": "Relative error",
        "percent": True,
        "log": True,
    },
    "median_detF_error": {
        "title": "Median relative det(F) error",
        "label": "Relative error",
        "percent": True,
        "log": True,
    },
    "median_true_energy_error": {
        "title": "Median relative true-energy error",
        "label": "Relative error",
        "percent": True,
        "log": True,
    },
    "median_quadrature_error": {
        "title": "Median reduced loss-vs-true energy error",
        "label": "Relative error",
        "percent": True,
        "log": True,
    },
    "raw_failure_rate": {
        "title": "Raw failure rate",
        "label": "Failure fraction",
        "percent": True,
        "log": False,
    },
    "acceptance_fraction": {
        "title": "Accepted trial fraction",
        "label": "Accepted fraction",
        "percent": True,
        "log": False,
    },
    "median_final_rep_atoms": {
        "title": "Median final representative-atom count",
        "label": "Representative atoms",
        "percent": False,
        "log": False,
    },
    "median_rep_atoms_spawned": {
        "title": "Median number of spawned representative atoms",
        "label": "Spawned representative atoms",
        "percent": False,
        "log": False,
    },
}


def format_annotation(value: float, percent: bool) -> str:
    if not np.isfinite(value):
        return "NA"

    if percent:
        return f"{100.0 * value:.2g}%"

    if abs(value) >= 100:
        return f"{value:.0f}"

    if abs(value) >= 10:
        return f"{value:.1f}"

    return f"{value:.2f}"


def make_heatmap(
    aggregate: pd.DataFrame,
    metric: str,
    reset: bool,
    output_dir: Path,
) -> None:
    spec = HEATMAP_SPECS[metric]

    subset = aggregate[aggregate["reset"] == reset]

    table = subset.pivot(
        index="interface_layers",
        columns="boundary_layers",
        values=metric,
    ).reindex(
        index=EXPECTED_INTERFACE_LAYERS,
        columns=EXPECTED_BOUNDARY_LAYERS,
    )

    raw_values = table.to_numpy(dtype=np.float64)
    plotted_values = raw_values.copy()

    if spec["percent"]:
        plotted_values *= 100.0

    finite = plotted_values[np.isfinite(plotted_values)]

    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    if len(finite) == 0:
        image = ax.imshow(
            np.zeros_like(plotted_values),
            origin="lower",
            aspect="auto",
        )
    else:
        norm = None

        if spec["log"]:
            positive = finite[finite > 0.0]

            if len(positive) > 0:
                vmin = positive.min()
                vmax = positive.max()

                if np.isclose(vmin, vmax):
                    vmax = vmin * 1.01

                norm = LogNorm(vmin=vmin, vmax=vmax)

                # LogNorm cannot display exactly zero. Mask zero/nonpositive
                # values rather than replacing them with arbitrary numbers.
                plotted_values = np.ma.masked_where(
                    plotted_values <= 0.0,
                    plotted_values,
                )

        image = ax.imshow(
            plotted_values,
            origin="lower",
            aspect="auto",
            norm=norm,
        )

    reset_label = "RESET" if reset else "non-RESET"

    ax.set_title(f"{spec['title']} — {reset_label}")
    ax.set_xlabel("Boundary layers")
    ax.set_ylabel("Interface layers")

    ax.set_xticks(np.arange(len(EXPECTED_BOUNDARY_LAYERS)))
    ax.set_xticklabels(EXPECTED_BOUNDARY_LAYERS)

    ax.set_yticks(np.arange(len(EXPECTED_INTERFACE_LAYERS)))
    ax.set_yticklabels(EXPECTED_INTERFACE_LAYERS)

    for row_index, il in enumerate(EXPECTED_INTERFACE_LAYERS):
        for col_index, bl in enumerate(EXPECTED_BOUNDARY_LAYERS):
            value = raw_values[row_index, col_index]

            ax.text(
                col_index,
                row_index,
                format_annotation(
                    value=value,
                    percent=spec["percent"],
                ),
                ha="center",
                va="center",
            )

    colorbar = fig.colorbar(image, ax=ax)

    if spec["percent"]:
        colorbar.set_label(spec["label"] + " (%)")
    else:
        colorbar.set_label(spec["label"])

    fig.tight_layout()

    reset_suffix = "reset" if reset else "nonreset"
    output_path = output_dir / f"heatmap_{metric}_{reset_suffix}.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")


def generate_all_heatmaps(
    aggregate: pd.DataFrame,
    output_dir: Path,
) -> None:
    for metric in HEATMAP_SPECS:
        if metric not in aggregate.columns:
            print(f"Skipping unavailable metric: {metric}")
            continue

        for reset in EXPECTED_RESET_VALUES:
            make_heatmap(
                aggregate=aggregate,
                metric=metric,
                reset=reset,
                output_dir=output_dir,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired full/reduced PINN histories and generate "
            "RESET/non-RESET parameter heatmaps."
        )
    )

    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=f"Directory containing JSON histories (default: {DEFAULT_HISTORY_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Analysis output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    return parser.parse_args()


def main() -> int:


    args = parse_args()

    history_dir = args.history_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"History directory: {history_dir}")
    print(f"Output directory:  {output_dir}")

    inventory = build_inventory(history_dir)
    inventory.to_csv(output_dir / "history_inventory.csv", index=False)

    print(f"\nFound {len(inventory)} matching JSON files.")

    validate_inventory(inventory)

    paired = pair_histories(inventory)
    paired.to_csv(output_dir / "paired_history_inventory.csv", index=False)

    print(f"Found {len(paired)} unique experiment keys.")

    atom_pos = make_triangular_lattice(
        nx=NX,
        ny=NY,
        spacing=LATTICE_SPACING,
    )

    lammps_timesteps, lammps_frames = read_lammps_trajectory_xy(
        LAMMPS_TRAJ_FILE
    )

    lammps_final_pos = np.asarray(
        lammps_frames[-1],
        dtype=np.float64,
    )

    summary = analyze_all_pairs(
        paired=paired,
        atom_pos=atom_pos,
        lammps_final_pos=lammps_final_pos,
    )


    # Detect unusually large final-energy-error trials.
    summary = mark_final_energy_outliers(summary)
    
    summary_path = output_dir / "trial_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")

    print(f"\nWrote {summary_path}")

    failures = summary[summary["failed"]]

    if not failures.empty:
        failures_path = output_dir / "failed_trials.csv"
        failures.to_csv(failures_path, index=False)

        print(f"Wrote {failures_path}")
        print("\nFailure counts:")
        print(
            failures["failure_reason"]
            .value_counts(dropna=False)
            .to_string()
        )

    energy_outliers = summary[
        (~summary["failed"])
        & summary["energy_outlier"]
    ].copy()

    if not energy_outliers.empty:
        outlier_path = (
            output_dir
            / "rejected_final_energy_outliers.csv"
        )

        energy_outliers.to_csv(
            outlier_path,
            index=False,
        )

        print(f"Wrote {outlier_path}")

        print("\nRejected final-energy outlier trials:")

        display_columns = [
            "interface_layers",
            "boundary_layers",
            "reset",
            "trial",
            "relative_true_energy_error",
            "energy_outlier_method",
            "energy_outlier_score",
            "energy_outlier_upper_limit",
        ]

        print(
            energy_outliers[display_columns]
            .sort_values(
                [
                    "reset",
                    "interface_layers",
                    "boundary_layers",
                    "trial",
                ]
            )
            .to_string(index=False)
        )

    aggregate = aggregate_trials(summary)

    aggregate_path = output_dir / "parameter_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print(f"Wrote {aggregate_path}")

    print("\nAggregated parameter summary:")
    print(
        aggregate.sort_values(
            ["reset", "interface_layers", "boundary_layers"]
        ).to_string(index=False)
    )

    generate_all_heatmaps(
        aggregate=aggregate,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
