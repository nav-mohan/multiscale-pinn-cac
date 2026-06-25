# Piecewise deformation-gradient discontinuity test.
#
# The deformation is imposed analytically and gradually ramped from the
# reference lattice to a piecewise-affine final configuration.
#
# JSON row format is unchanged:
#     epoch
#     loss_energy
#     true_energy
#     pos
#     rep_indices
#     q_full

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

PINN_HISTORY_JSON_FILENAME = (
    "pinn-history-2d-piecewise/pinn-history-2d-piecewise-defgrad.json"
)

MIN_EPOCHS_BEFORE_RECLUSTER = 1000
RECLUSTER_EVERY = 500

DEFGRAD_TRIGGER_TOL = 1.0e-3
DEFGRAD_JUMP_TOL = 5.0e-3
DEFGRAD_JUMP_MAD_FACTOR = 3.0

MIN_SPLIT_CLUSTER_SIZE = 10
MIN_DISCONTINUITY_EDGES = 3


# Target deformation gradients.
#
# Their second columns are identical. Therefore, the deformation remains
# continuous along the vertical interface even though F is discontinuous.

F_LEFT_FINAL = np.array(
    [
        [1.02, 0.02],
        [0.00, 0.99],
    ],
    dtype=np.float64,
)

F_RIGHT_FINAL = np.array(
    [
        [1.08, 0.02],
        [0.00, 0.99],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------
# Piecewise deformation
# ---------------------------------------------------------------------

def make_piecewise_deformed_positions(
    reference_pos,
    interface_x,
    F_left,
    F_right,
):
    """
    Apply a continuous piecewise-affine deformation separated by

        X[0] = interface_x.

    The mapping is

        x = F_left X                         on the left
        x = F_right X + translation         on the right

    The translation makes the deformation continuous at the interface.

    For continuity along the complete vertical interface, F_left and
    F_right must have identical second columns.
    """
    reference_pos = np.asarray(reference_pos, dtype=np.float64)
    F_left = np.asarray(F_left, dtype=np.float64)
    F_right = np.asarray(F_right, dtype=np.float64)

    if F_left.shape != (2, 2) or F_right.shape != (2, 2):
        raise ValueError("F_left and F_right must both have shape (2, 2).")

    if not np.allclose(F_left[:, 1], F_right[:, 1]):
        raise ValueError(
            "For a continuous vertical interface, F_left and F_right "
            "must have identical second columns."
        )

    left_mask = reference_pos[:, 0] <= interface_x
    right_mask = ~left_mask

    current_pos = np.empty_like(reference_pos)

    current_pos[left_mask] = (
        reference_pos[left_mask] @ F_left.T
    )

    interface_point = np.array(
        [interface_x, 0.0],
        dtype=np.float64,
    )

    translation = (
        F_left @ interface_point
        - F_right @ interface_point
    )

    current_pos[right_mask] = (
        reference_pos[right_mask] @ F_right.T
        + translation
    )

    return current_pos


def deformation_gradients_at_load(load_fraction):
    """
    Linearly ramp each deformation gradient from identity to its
    prescribed final value.
    """
    load_fraction = float(
        np.clip(load_fraction, 0.0, 1.0)
    )

    identity = np.eye(2)

    F_left = (
        identity
        + load_fraction * (F_LEFT_FINAL - identity)
    )

    F_right = (
        identity
        + load_fraction * (F_RIGHT_FINAL - identity)
    )

    return F_left, F_right


# ---------------------------------------------------------------------
# Lattice and atomistic/continuum partition
# ---------------------------------------------------------------------

atom_pos = make_triangular_lattice(
    NX,
    NY,
    LATTICE_SPACING,
)

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

interface_x = 0.5 * (
    atom_pos[:, 0].min()
    + atom_pos[:, 0].max()
)

print("interface x:", interface_x)
print("number of atoms:", len(atom_pos))
print("atomistic atoms:", len(atomistic_indices))
print("continuum atoms:", len(continuum_indices))


# ---------------------------------------------------------------------
# Reference neighbor lists
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Initial clustering
# ---------------------------------------------------------------------

initial_rep_indices, rep_targets = pick_manual_rep_atoms(
    atom_pos,
    continuum_indices,
    0.1,
)

# Use uniform importance initially. In the undeformed configuration,
# q_i = ||F_i - I|| is approximately zero everywhere.
initial_importance = np.ones(
    len(continuum_indices),
    dtype=np.float64,
)

(
    cluster_idx,
    centers,
    rep_indices,
    cac_weights,
    kmeans_history,
) = kmeans_weighted(
    atom_pos=atom_pos,
    continuum_indices=continuum_indices,
    initial_rep_indices=initial_rep_indices,
    importance_weights=initial_importance,
)

print("initial representative indices:", initial_rep_indices)
print("final initial representatives:", rep_indices)
print("initial CAC weights:", cac_weights)
print("sum of CAC weights:", cac_weights.sum())


# ---------------------------------------------------------------------
# Fixed interaction-pair list
# ---------------------------------------------------------------------

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

atomistic_indices_t = torch.tensor(
    atomistic_indices,
    dtype=TORCH_LONG,
    device=TORCH_DEVICE,
)


# ---------------------------------------------------------------------
# Energy calculation
# ---------------------------------------------------------------------

def calculate_full_and_reduced_energy(
    current_pos_np,
    rep_indices,
    cac_weights,
):
    """
    Compute the clustered energy and full all-atom energy for the
    prescribed configuration.
    """
    current_pos_t = torch.tensor(current_pos_np,dtype=TORCH_FLOAT,device=TORCH_DEVICE,)

    rep_indices_t = torch.tensor(rep_indices,dtype=TORCH_LONG,device=TORCH_DEVICE,)

    cac_weights_t = torch.tensor(cac_weights,dtype=TORCH_FLOAT,device=TORCH_DEVICE,)

    with torch.no_grad():
        site_E = site_energies_from_pair_list(current_pos_t,pair_i,pair_j,)

        atomistic_energy = site_E[atomistic_indices_t].sum()

        cluster_energy = torch.sum(cac_weights_t * site_E[rep_indices_t])

        reduced_energy = (atomistic_energy + cluster_energy)

        full_energy = site_E.sum()

    return (
        float(reduced_energy.cpu()),
        float(full_energy.cpu()),
    )


# ---------------------------------------------------------------------
# Controlled piecewise-deformation experiment
# ---------------------------------------------------------------------

def run_piecewise_deformation_test():
    global cluster_idx
    global centers
    global rep_indices
    global cac_weights

    output_path = Path(PINN_HISTORY_JSON_FILENAME)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_epochs = list(range(0, MAX_ITER, LOG_EVERY))

    if frame_epochs[-1] != MAX_ITER - 1:
        frame_epochs.append(MAX_ITER - 1)

    with output_path.open("w") as jsonfile:
        jsonfile.write("[\n")
        first_row = True

        for epoch in frame_epochs:
            load_fraction = epoch / max(MAX_ITER - 1, 1)

            F_left, F_right = deformation_gradients_at_load(load_fraction)

            current_pos_np = make_piecewise_deformed_positions(reference_pos=atom_pos,interface_x=interface_x,F_left=F_left,F_right=F_right,)

            (F_full,q_defgrad_full,valid_full,) = local_deformation_gradient(reference_pos=atom_pos,current_pos=current_pos_np,neighs=neighs,)

            should_check_reclustering = (epoch >= MIN_EPOCHS_BEFORE_RECLUSTER and epoch % RECLUSTER_EVERY == 0)

            if should_check_reclustering:
                did_recluster, recluster_info = (
                    recluster_deformation_gradient_weighted(
                        atom_pos_ref=atom_pos,
                        current_pos_np=current_pos_np,
                        continuum_indices=continuum_indices,
                        cluster_idx=cluster_idx,
                        rep_indices=rep_indices,
                        cac_weights=cac_weights,
                        neighs=neighs,
                        trigger_tol=DEFGRAD_TRIGGER_TOL,
                        jump_tol=DEFGRAD_JUMP_TOL,
                        jump_mad_factor=(
                            DEFGRAD_JUMP_MAD_FACTOR
                        ),
                        min_split_cluster_size=(
                            MIN_SPLIT_CLUSTER_SIZE
                        ),
                        min_cut_edges=(
                            MIN_DISCONTINUITY_EDGES
                        ),
                        max_iter=100,
                        tol=1.0e-10,
                    )
                )

                print(
                    "epoch:",
                    epoch,
                    "load:",
                    f"{load_fraction:.4f}",
                    "quadrature error:",
                    recluster_info[
                        "rel_defgrad_quad_error"
                    ],
                )

                if did_recluster:
                    split_info = recluster_info[
                        "split_info"
                    ]

                    print(
                        "quadrature triggered:",
                        recluster_info[
                            "quadrature_triggered"
                        ],
                    )
                    print(
                        "discontinuity triggered:",
                        recluster_info[
                            "discontinuity_triggered"
                        ],
                    )
                    print(
                        "clusters:",
                        split_info[
                            "num_clusters_before"
                        ],
                        "->",
                        split_info[
                            "num_clusters_after"
                        ],
                    )

                    for split in split_info["splits"]:
                        print("split:", split)

                    cluster_idx = recluster_info[
                        "cluster_idx"
                    ]
                    centers = recluster_info[
                        "centers"
                    ]
                    rep_indices = recluster_info[
                        "rep_indices"
                    ]
                    cac_weights = recluster_info[
                        "cac_weights"
                    ]

                    assert len(cluster_idx) == len(
                        continuum_indices
                    )
                    assert len(rep_indices) == len(
                        cac_weights
                    )
                    assert np.isclose(
                        cac_weights.sum(),
                        len(continuum_indices),
                    )
                    assert set(rep_indices).issubset(
                        set(continuum_indices)
                    )

                    print(
                        "new representative indices:",
                        rep_indices,
                    )
                    print(
                        "new CAC weights:",
                        cac_weights,
                    )

            # Calculate energy after any reclustering so that positions,
            # representatives and energy belong to the same frame.
            loss_energy, true_energy = (
                calculate_full_and_reduced_energy(
                    current_pos_np=current_pos_np,
                    rep_indices=rep_indices,
                    cac_weights=cac_weights,
                )
            )

            # Preserve exactly the JSON row format used previously.
            row = {
                "epoch": int(epoch),
                "loss_energy": loss_energy,
                "true_energy": true_energy,
                "pos": current_pos_np.tolist(),
                "rep_indices": rep_indices.tolist(),
                "q_full": q_defgrad_full.tolist(),
            }

            if not first_row:
                jsonfile.write(",\n")

            jsonfile.write(json.dumps(row))
            first_row = False
            jsonfile.flush()

            print(
                epoch,
                "load_fraction:",
                f"{load_fraction:.4f}",
                "reduced energy:",
                f"{loss_energy:.8e}",
                "true energy:",
                f"{true_energy:.8e}",
                "representatives:",
                len(rep_indices),
                "q max:",
                f"{q_defgrad_full.max():.8e}",
            )

        jsonfile.write("\n]\n")

    print("Saved history to:", output_path)


def main():
    print("Final prescribed deformation-gradient jump:")
    print(F_LEFT_FINAL - F_RIGHT_FINAL)
    print(
        "Frobenius jump magnitude:",
        np.linalg.norm(
            F_LEFT_FINAL - F_RIGHT_FINAL,
            ord="fro",
        ),
    )

    run_piecewise_deformation_test()


if __name__ == "__main__":
    main()
