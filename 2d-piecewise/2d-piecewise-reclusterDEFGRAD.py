# PINN training toward a prescribed piecewise-affine target.
# Reclustering is triggered by the measured deformation-gradient jump.
# New representative atoms are spawned near under-resolved discontinuities.

import json
from pathlib import Path

import numpy as np
import torch

from shared_2D import *


# ---------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------

NX, NY = 49, 50
LATTICE_SPACING = 2 ** (1 / 6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9

MAX_ITER = 10000
LOG_EVERY = 100

PINN_HISTORY_JSON_FILENAME = "pinn-history-2d-piecewise/pinn-history-2d-piecewise-reclusterDEFGRAD.json"
MIN_EPOCHS_BEFORE_RECLUSTER = 1000
RECLUSTER_EVERY = 500
DEFGRAD_JUMP_TOL = 5.0e-3
DEFGRAD_JUMP_MAD_FACTOR = 3.0
TARGET_PENALTY_STIFFNESS = 50.0
LOAD_RAMP_EPOCHS = 3000
MAX_TOTAL_REP_ATOMS = 50
MAX_NEW_REPS_PER_EVENT = 5
ATOMS_PER_NEW_REP = 10
JUMP_IMPORTANCE_WEIGHT = 10.0
REP_INTERFACE_SPACING = 1.5 * LATTICE_SPACING



# ---------------------------------------------------------------------
# Deformation-gradient diagnostics
# ---------------------------------------------------------------------

def deformation_gradient_jump_indicator(
    F_full,
    valid_full,
    neighs,
    continuum_indices,
):
    """
    Compute

        J_i = max_{j in N(i)} ||F_i - F_j||_F

    using valid continuum neighbors.
    """
    continuum_indices = np.asarray(
        continuum_indices,
        dtype=int,
    )
    continuum_set = set(continuum_indices.tolist())

    jump_full = np.zeros(
        len(F_full),
        dtype=np.float64,
    )

    for global_i in continuum_indices:
        if not valid_full[global_i]:
            continue

        for global_j in neighs[global_i]:
            global_j = int(global_j)

            if global_j not in continuum_set:
                continue

            if not valid_full[global_j]:
                continue

            jump = np.linalg.norm(
                F_full[global_i] - F_full[global_j],
                ord="fro",
            )

            jump_full[global_i] = max(
                jump_full[global_i],
                jump,
            )

    return jump_full


def effective_jump_threshold(
    jump_full,
    continuum_indices,
    absolute_tol,
    mad_factor,
):
    """
    Combine an absolute threshold with a robust MAD-based threshold.
    """
    jump_values = np.asarray(
        jump_full[continuum_indices],
        dtype=np.float64,
    )

    positive_jumps = jump_values[jump_values > 0.0]

    if len(positive_jumps) == 0:
        adaptive_tol = 0.0
        median_jump = 0.0
        mad_jump = 0.0
    else:
        median_jump = np.median(positive_jumps)
        mad_jump = np.median(
            np.abs(positive_jumps - median_jump)
        )

        adaptive_tol = (
            median_jump
            + mad_factor * 1.4826 * mad_jump
        )

    threshold = max(
        float(absolute_tol),
        float(adaptive_tol),
    )

    return threshold, median_jump, mad_jump


# ---------------------------------------------------------------------
# Representative spawning and importance
# ---------------------------------------------------------------------

def spawn_representatives_near_discontinuity(
    reference_pos,
    continuum_indices,
    rep_indices,
    jump_full,
    jump_tol,
    max_new_reps=10,
    atoms_per_new_rep=10,
    min_separation=None,
):
    """
    Spawn representative atoms near an under-resolved discontinuity.

    Candidates are ranked using jump-weighted farthest-point sampling.
    """
    reference_pos = np.asarray(
        reference_pos,
        dtype=np.float64,
    )
    continuum_indices = np.asarray(
        continuum_indices,
        dtype=int,
    )
    rep_indices = np.asarray(
        rep_indices,
        dtype=int,
    )

    existing_reps = set(rep_indices.tolist())

    candidate_global = np.asarray(
        [
            global_idx
            for global_idx in continuum_indices
            if (
                jump_full[global_idx] > jump_tol
                and global_idx not in existing_reps
            )
        ],
        dtype=int,
    )

    if len(candidate_global) == 0 or max_new_reps <= 0:
        return rep_indices.copy(), np.empty(0, dtype=int)

    n_new = int(
        np.ceil(
            len(candidate_global) / atoms_per_new_rep
        )
    )
    n_new = min(n_new, max_new_reps)

    candidate_pos = reference_pos[candidate_global]
    candidate_jump = jump_full[candidate_global]

    selected = []
    available = np.ones(
        len(candidate_global),
        dtype=bool,
    )

    for _ in range(n_new):
        if selected:
            current_rep_indices = np.concatenate(
                [
                    rep_indices,
                    np.asarray(selected, dtype=int),
                ]
            )
        else:
            current_rep_indices = rep_indices

        current_rep_pos = reference_pos[
            current_rep_indices
        ]

        distance_to_rep = np.min(
            np.linalg.norm(
                candidate_pos[:, None, :]
                - current_rep_pos[None, :, :],
                axis=2,
            ),
            axis=1,
        )

        normalized_jump = candidate_jump / (
            candidate_jump.max() + 1.0e-12
        )

        score = normalized_jump * distance_to_rep
        score[~available] = -np.inf

        best_local = int(np.argmax(score))

        if not np.isfinite(score[best_local]):
            break

        if (
            min_separation is not None
            and distance_to_rep[best_local]
            < min_separation
        ):
            break

        best_global = int(
            candidate_global[best_local]
        )

        selected.append(best_global)
        available[best_local] = False

    spawned = np.asarray(selected, dtype=int)

    expanded_rep_indices = np.concatenate(
        [rep_indices, spawned]
    )

    return expanded_rep_indices, spawned


def make_adaptive_defgrad_importance(
    q_defgrad_full,
    jump_full,
    continuum_indices,
    jump_weight=10.0,
    eps=1.0e-12,
):
    """
    Construct positive K-means importance weights from deformation
    magnitude and deformation-gradient jump.
    """
    q = np.asarray(
        q_defgrad_full[continuum_indices],
        dtype=np.float64,
    )

    jump = np.asarray(
        jump_full[continuum_indices],
        dtype=np.float64,
    )

    positive_q = q[q > eps]
    positive_jump = jump[jump > eps]

    q_scale = (
        np.percentile(positive_q, 95)
        if len(positive_q) > 0
        else 1.0
    )

    jump_scale = (
        np.percentile(positive_jump, 95)
        if len(positive_jump) > 0
        else 1.0
    )

    q_normalized = q / (q_scale + eps)
    jump_normalized = jump / (
        jump_scale + eps
    )

    importance = (
        1.0
        + q_normalized
        + jump_weight * jump_normalized
    )

    return np.clip(
        importance,
        0.1,
        100.0,
    )


# ---------------------------------------------------------------------
# Piecewise-affine target deformation
# ---------------------------------------------------------------------

def make_piecewise_deformed_positions(
    reference_pos,
    interface_x,
    F_left,
    F_right,
):
    """
    Construct a continuous piecewise-affine deformation separated by
    the vertical interface X[0] = interface_x.

    Continuity along the full interface requires F_left and F_right
    to have identical second columns.
    """
    reference_pos = np.asarray(
        reference_pos,
        dtype=np.float64,
    )
    F_left = np.asarray(F_left, dtype=np.float64)
    F_right = np.asarray(F_right, dtype=np.float64)

    if F_left.shape != (2, 2):
        raise ValueError("F_left must have shape (2, 2).")

    if F_right.shape != (2, 2):
        raise ValueError("F_right must have shape (2, 2).")

    if not np.allclose(
        F_left[:, 1],
        F_right[:, 1],
    ):
        raise ValueError(
            "F_left and F_right must have identical "
            "second columns for continuity along a "
            "vertical interface."
        )

    interface_point = np.array(
        [interface_x, 0.0],
        dtype=np.float64,
    )

    left = reference_pos[:, 0] <= interface_x
    right = ~left

    current_pos = np.empty_like(reference_pos)

    current_pos[left] = (
        reference_pos[left] @ F_left.T
    )

    translation = (
        F_left @ interface_point
        - F_right @ interface_point
    )

    current_pos[right] = (
        reference_pos[right] @ F_right.T
        + translation
    )

    return current_pos


def make_target_displacement(epoch):
    """
    Ramp the piecewise target from the identity deformation to its
    prescribed final value.
    """
    load_fraction = min(
        float(epoch) / float(LOAD_RAMP_EPOCHS),
        1.0,
    )

    identity = np.eye(2)

    F_left_epoch = (
        identity
        + load_fraction * (F_LEFT - identity)
    )

    F_right_epoch = (
        identity
        + load_fraction * (F_RIGHT - identity)
    )

    target_pos = make_piecewise_deformed_positions(
        reference_pos=atom_pos,
        interface_x=INTERFACE_X,
        F_left=F_left_epoch,
        F_right=F_right_epoch,
    )

    target_u = torch.tensor(
        target_pos - atom_pos,
        dtype=TORCH_FLOAT,
        device=TORCH_DEVICE,
    )

    return target_u, load_fraction


# ---------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------

atom_pos = make_triangular_lattice(
    NX,
    NY,
    LATTICE_SPACING,
)

INTERFACE_X = 0.5 * (
    atom_pos[:, 0].min()
    + atom_pos[:, 0].max()
)

# this has shear
# F_LEFT = np.array([
#   [1.02, 0.02],
#   [0.00, 0.99],
#])
# F_RIGHT = np.array([
#   [1.08, 0.02],
#   [0.00, 0.99],
#])

# this has no shear. 
# F_LEFT = np.array([
#     [1.02, 0.00],
#     [0.00, 0.99],
# ])
# F_RIGHT = np.array([
#     [1.08, 0.00],
#     [0.00, 0.99],
# ])

# this has no shear, oppsite directions. 
F_LEFT = np.array([
    [1.02, 0.00],
    [0.00, 0.00],
])
F_RIGHT = np.array([
    [0.97, 0.00],
    [0.00, 0.00],
])


# ---------------------------------------------------------------------
# Atomistic/continuum partition and initial clustering
# ---------------------------------------------------------------------

atomistic_indices, continuum_indices = (
    split_atomistic_continuum(
        NX,
        NY,
        BORDER_LAYERS,
    )
)

atomistic_indices = np.asarray(
    atomistic_indices,
    dtype=int,
)
continuum_indices = np.asarray(
    continuum_indices,
    dtype=int,
)

initial_rep_indices, rep_targets = (
    pick_manual_rep_atoms_piecewise(
        atom_pos,
        continuum_indices,
        0.1,
    )
)

neighs = build_first_neighbor_list(
    atom_pos,
    spacing=LATTICE_SPACING,
)

neighbor_counts = np.asarray(
    [len(atom_neighbors) for atom_neighbors in neighs]
)

print(
    "neighbor counts:",
    np.unique(
        neighbor_counts,
        return_counts=True,
    ),
)
print("maximum neighbors:", neighbor_counts.max())

# The undeformed initial state has q approximately zero everywhere.
# Uniform importance avoids arbitrary weighting by numerical noise.
initial_importance = np.ones(
    len(continuum_indices),
    dtype=np.float64,
)

(
    cluster_idx,
    centers,
    rep_indices,
    cac_weights,
    history,
) = kmeans_weighted(
    atom_pos=atom_pos,
    continuum_indices=continuum_indices,
    initial_rep_indices=initial_rep_indices,
    importance_weights=initial_importance,
)

print("number of atoms:", len(atom_pos))
print("atomistic atoms:", len(atomistic_indices))
print("continuum atoms:", len(continuum_indices))
print("initial representatives:", initial_rep_indices)
print("final representatives:", rep_indices)
print("CAC weights:", cac_weights)
print("sum of CAC weights:", cac_weights.sum())


# ---------------------------------------------------------------------
# Torch setup
# ---------------------------------------------------------------------

r0, r0_norm = generate_normalized_input(
    atom_pos=atom_pos,
)

# The distributed target penalty removes rigid-body modes.
mask = torch.ones_like(r0)
u_prescribed = torch.zeros_like(r0)

pair_i_np, pair_j_np = build_reference_pair_list(
    atom_pos,
    cutoff=CUTOFF_FACTOR * LATTICE_SPACING,
)

pair_i = torch.tensor(
    pair_i_np,
    dtype=TORCH_LONG,
    device=TORCH_DEVICE,
)
pair_j = torch.tensor(
    pair_j_np,
    dtype=TORCH_LONG,
    device=TORCH_DEVICE,
)

model = PINN2D().to(
    device=TORCH_DEVICE,
    dtype=TORCH_FLOAT,
)

adam_opt = torch.optim.Adam(
    model.parameters(),
    lr=1.0e-4,
)

atomistic_indices_t = torch.tensor(
    atomistic_indices,
    dtype=TORCH_LONG,
    device=TORCH_DEVICE,
)
rep_indices_t = torch.tensor(
    rep_indices,
    dtype=TORCH_LONG,
    device=TORCH_DEVICE,
)
cac_weights_t = torch.tensor(
    cac_weights,
    dtype=TORCH_FLOAT,
    device=TORCH_DEVICE,
)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def do_train_piecewise_recluster_defgrad():
    global cluster_idx
    global centers
    global rep_indices
    global cac_weights
    global rep_indices_t
    global cac_weights_t
    global adam_opt

    output_path = Path(
        PINN_HISTORY_JSON_FILENAME
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as jsonfile:
        jsonfile.write("[\n")
        first_row = True

        for epoch in range(MAX_ITER):
            adam_opt.zero_grad()

            target_u, load_fraction = (
                make_target_displacement(epoch)
            )

            u_raw = model(r0_norm)
            u = mask * u_raw + u_prescribed
            pos = r0 + u

            site_E = site_energies_from_pair_list(
                pos,
                pair_i,
                pair_j,
            )

            atomistic_energy = site_E[
                atomistic_indices_t
            ].sum()

            cluster_energy = torch.sum(
                cac_weights_t
                * site_E[rep_indices_t]
            )

            target_penalty = (
                0.5
                * TARGET_PENALTY_STIFFNESS
                * torch.sum((u - target_u) ** 2)
            )

            loss_energy = (
                atomistic_energy
                + cluster_energy
                + target_penalty
            )

            loss_energy.backward()
            adam_opt.step()

            should_check_reclustering = (
                epoch >= MIN_EPOCHS_BEFORE_RECLUSTER
                and epoch % RECLUSTER_EVERY == 0
            )

            if should_check_reclustering:
                with torch.no_grad():
                    u_eval = (
                        mask * model(r0_norm)
                        + u_prescribed
                    )
                    pos_eval = r0 + u_eval

                current_pos_np = (
                    pos_eval.detach().cpu().numpy()
                )

                F_full, q_full, valid_full = (
                    local_deformation_gradient(
                        reference_pos=atom_pos,
                        current_pos=current_pos_np,
                        neighs=neighs,
                    )
                )

                jump_full = (
                    deformation_gradient_jump_indicator(
                        F_full=F_full,
                        valid_full=valid_full,
                        neighs=neighs,
                        continuum_indices=(
                            continuum_indices
                        ),
                    )
                )

                (
                    effective_jump_tol,
                    median_jump,
                    mad_jump,
                ) = effective_jump_threshold(
                    jump_full=jump_full,
                    continuum_indices=continuum_indices,
                    absolute_tol=DEFGRAD_JUMP_TOL,
                    mad_factor=(
                        DEFGRAD_JUMP_MAD_FACTOR
                    ),
                )

                jump_candidates = continuum_indices[
                    jump_full[continuum_indices]
                    > effective_jump_tol
                ]

                spawned = np.empty(0, dtype=int)

                if len(jump_candidates) > 0:
                    distance_to_rep = np.min(
                        np.linalg.norm(
                            atom_pos[
                                jump_candidates,
                                None,
                                :,
                            ]
                            - atom_pos[
                                rep_indices
                            ][None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )

                    maximum_unresolved_distance = (
                        distance_to_rep.max()
                    )

                    interface_underresolved = (
                        maximum_unresolved_distance
                        > REP_INTERFACE_SPACING
                    )

                    remaining_capacity = max(
                        MAX_TOTAL_REP_ATOMS
                        - len(rep_indices),
                        0,
                    )

                    if (
                        interface_underresolved
                        and remaining_capacity > 0
                    ):
                        (
                            expanded_reps,
                            spawned,
                        ) = (
                            spawn_representatives_near_discontinuity(
                                reference_pos=atom_pos,
                                continuum_indices=(
                                    continuum_indices
                                ),
                                rep_indices=rep_indices,
                                jump_full=jump_full,
                                jump_tol=(
                                    effective_jump_tol
                                ),
                                max_new_reps=min(
                                    MAX_NEW_REPS_PER_EVENT,
                                    remaining_capacity,
                                ),
                                atoms_per_new_rep=(
                                    ATOMS_PER_NEW_REP
                                ),
                                min_separation=(
                                    REP_INTERFACE_SPACING
                                ),
                            )
                        )

                if len(spawned) > 0:
                    importance = (
                        make_adaptive_defgrad_importance(
                            q_defgrad_full=q_full,
                            jump_full=jump_full,
                            continuum_indices=(
                                continuum_indices
                            ),
                            jump_weight=(
                                JUMP_IMPORTANCE_WEIGHT
                            ),
                        )
                    )

                    (
                        cluster_idx,
                        centers,
                        rep_indices,
                        cac_weights,
                        history,
                    ) = kmeans_weighted(
                        # K-means does not know INTERFACE_X.
                        atom_pos=atom_pos,
                        continuum_indices=continuum_indices,
                        initial_rep_indices=expanded_reps,
                        importance_weights=importance,
                        max_iter=100,
                        tol=1.0e-10,
                    )

                    assert len(cluster_idx) == len(
                        continuum_indices
                    )
                    assert len(rep_indices) == len(
                        cac_weights
                    )
                    assert len(np.unique(rep_indices)) == len(
                        rep_indices
                    )
                    assert np.isclose(
                        cac_weights.sum(),
                        len(continuum_indices),
                    )
                    assert set(rep_indices).issubset(
                        set(continuum_indices)
                    )

                    rep_indices_t = torch.tensor(
                        rep_indices,
                        dtype=TORCH_LONG,
                        device=TORCH_DEVICE,
                    )
                    cac_weights_t = torch.tensor(
                        cac_weights,
                        dtype=TORCH_FLOAT,
                        device=TORCH_DEVICE,
                    )

                    # The reduced loss changed after reclustering.
                    adam_opt = torch.optim.Adam(
                        model.parameters(),
                        lr=1.0e-4,
                    )

                    print("epoch:", epoch)
                    print(
                        "effective jump threshold:",
                        effective_jump_tol,
                    )
                    print(
                        "median/MAD:",
                        median_jump,
                        mad_jump,
                    )
                    print(
                        "spawned representatives:",
                        spawned,
                    )
                    print(
                        "representative count:",
                        len(rep_indices),
                    )
                    print(
                        "sum of CAC weights:",
                        cac_weights.sum(),
                    )

            # ---------------------------------------------------------
            # Log a consistent post-step, post-reclustering frame.
            # ---------------------------------------------------------

            if epoch % LOG_EVERY == 0:
                with torch.no_grad():
                    target_u_log, _ = (
                        make_target_displacement(epoch)
                    )

                    u_log = (
                        mask * model(r0_norm)
                        + u_prescribed
                    )
                    pos_log = r0 + u_log

                    site_E_log = (
                        site_energies_from_pair_list(
                            pos_log,
                            pair_i,
                            pair_j,
                        )
                    )

                    atomistic_energy_log = (
                        site_E_log[
                            atomistic_indices_t
                        ].sum()
                    )

                    cluster_energy_log = torch.sum(
                        cac_weights_t
                        * site_E_log[rep_indices_t]
                    )

                    target_penalty_log = (
                        0.5
                        * TARGET_PENALTY_STIFFNESS
                        * torch.sum(
                            (u_log - target_u_log) ** 2
                        )
                    )

                    loss_energy_log = (
                        atomistic_energy_log
                        + cluster_energy_log
                        + target_penalty_log
                    )

                    true_energy_log = (
                        site_E_log.sum()
                        + target_penalty_log
                    )

                    target_rmse = torch.sqrt(
                        torch.mean(
                            (u_log - target_u_log) ** 2
                        )
                    )

                current_pos_np = (
                    pos_log.detach().cpu().numpy()
                )

                _, q_full_log, _ = (
                    local_deformation_gradient(
                        reference_pos=atom_pos,
                        current_pos=current_pos_np,
                        neighs=neighs,
                    )
                )

                # Preserve the original JSON row format.
                row = {
                    "epoch": int(epoch),
                    "loss_energy": (
                        loss_energy_log.cpu().item()
                    ),
                    "true_energy": (
                        true_energy_log.cpu().item()
                    ),
                    "pos": current_pos_np.tolist(),
                    "rep_indices": (
                        rep_indices.tolist()
                    ),
                    "q_full": q_full_log.tolist(),
                }

                if not first_row:
                    jsonfile.write(",\n")

                jsonfile.write(json.dumps(row))
                first_row = False
                jsonfile.flush()

                print(
                    epoch,
                    "load:",
                    f"{load_fraction:.4f}",
                    "loss:",
                    f"{float(loss_energy_log):.8e}",
                    "target RMSE:",
                    f"{float(target_rmse):.8e}",
                    "representatives:",
                    len(rep_indices),
                )

        jsonfile.write("\n]\n")

    print("Saved history to:", output_path)


def main():
    do_train_piecewise_recluster_defgrad()


if __name__ == "__main__":
    main()