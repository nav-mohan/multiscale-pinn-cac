# kmeans weighted by deformation-gradient 
# reclustering triggered by variance of deformation-gradiet

from shared_2D import *

# problem specification
NX, NY = 49, 50
LATTICE_SPACING = 2**(1/6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9
MAX_ITER = 20000
LOG_EVERY = 100
PINN_HISTORY_JSON_FILENAME = "pinn_history/pinn-history-2d-neumann-reclusterDEFGRAD.json"
MIN_EPOCHS_BEFORE_RECLUSTER = 1000
RECLUSTER_EVERY = 500
DEFGRAD_TRIGGER_TOL = 1.0e-3
DEFGRAD_JUMP_TOL = 1.0e-3
DEFGRAD_JUMP_MAD_FACTOR = 4.0
MIN_SPLIT_CLUSTER_SIZE = 10
MIN_DISCONTINUITY_EDGES = 3
FORCE_PER_ATOM = np.array([0.5, 0.0])


atom_pos = make_triangular_lattice(NX,NY,LATTICE_SPACING)

# 3 layers of atoms bordering the square are flagged as atomistic-region
atomistic_indices,continuum_indices = split_atomistic_continuum(NX,NY,BORDER_LAYERS)
continuum_atom_pos = atom_pos[continuum_indices]
atomistic_atom_pos = atom_pos[atomistic_indices]

# manually pick 5 rep_atoms among the continuum atoms 
initial_rep_indices, rep_targets = pick_manual_rep_atoms(atom_pos,continuum_indices,0.1)

neighs = build_first_neighbor_list(atom_pos, spacing=LATTICE_SPACING)

neighbor_counts = np.array([len(nbrs) for nbrs in neighs])
print("neighbor counts:", np.unique(neighbor_counts, return_counts=True))
print("max neighbors:", neighbor_counts.max())

# compute deformation-gradient. this is approx 0 because we havent done any deformation yet
F_full, q_defgrad_full, valid_full = local_deformation_gradient(reference_pos=atom_pos,current_pos=atom_pos,neighs=neighs,)
q_continuum = q_defgrad_full[continuum_indices] # this is just an array of 1's
# assign continuum-atoms to their clusters, and equilibrate over 100 iterations of kmeans 
cluster_idx, centers, rep_indices, cac_weights, history = kmeans_weighted(
    atom_pos=atom_pos,
    continuum_indices=continuum_indices,
    initial_rep_indices=initial_rep_indices, 
    importance_weights=q_continuum
    )

print("number of atoms \t\t", len(atom_pos))
print("atomistic atoms \t\t", len(atomistic_indices))
print("continuum atoms \t\t", len(continuum_indices))
print("initial rep atom indices \t", initial_rep_indices)
print("final rep atom indices \t\t", rep_indices)
print("cac_weights \t\t\t", cac_weights)
print("sum of cac_weights \t\t\t", cac_weights.sum())


# normalized inputs
r0,r0_norm = generate_normalized_input(atom_pos=atom_pos)

# mask for applying dirichlet boundary condition. this is a very specific Dirichlet BC
mask,u_prescribed = generate_neumann_support_mask(r0=r0, atom_pos=atom_pos)

### Set up the neighbrlist once:
pair_i_np, pair_j_np = build_reference_pair_list(atom_pos, cutoff = CUTOFF_FACTOR*LATTICE_SPACING)
pair_i = torch.tensor(pair_i_np, dtype=TORCH_LONG, device=TORCH_DEVICE)
pair_j = torch.tensor(pair_j_np, dtype=TORCH_LONG, device=TORCH_DEVICE)

# initialize the PINN 
model = PINN2D().to(device=TORCH_DEVICE, dtype=TORCH_FLOAT)
adam_opt = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer adjusts the weights of the NN


atomistic_indices_t = torch.tensor(atomistic_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
rep_indices_t = torch.tensor(rep_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
cac_weights_t = torch.tensor(cac_weights, dtype=TORCH_FLOAT, device=TORCH_DEVICE)




# apply force-per-atom to atoms on the right boundary wall
# band-width determines how many atoms deep on the right-wall we will apply the force. we want to apply the force on two layers of atoms
def make_force_bc_right_boundary_band(atom_pos,force_per_atom,band_width):
    f_ext = np.zeros_like(atom_pos)

    xmax = atom_pos[:, 0].max()

    right_boundary_band = atom_pos[:, 0] >= xmax - band_width

    f_ext[right_boundary_band, :] = force_per_atom

    return f_ext, right_boundary_band

# initialize the array of external force-per-atom
f_ext_np, right_boundary_band = make_force_bc_right_boundary_band(atom_pos,force_per_atom=FORCE_PER_ATOM,band_width=0.6 * LATTICE_SPACING,)
print("number of atoms under force:", np.sum(right_boundary_band))
f_ext = torch.tensor(f_ext_np, dtype=TORCH_FLOAT, device=TORCH_DEVICE)

import json

def do_train_neuman_reclusterDEFGRAD():
    global cluster_idx
    global rep_indices
    global cac_weights
    global rep_indices_t
    global cac_weights_t
    global adam_opt
    global centers

    with open(PINN_HISTORY_JSON_FILENAME, "w") as jsonfile:
        jsonfile.write("[\n")
        first_row = True

        for epoch in range(MAX_ITER):
            adam_opt.zero_grad()

            u_raw = model(r0_norm)
            u = mask * u_raw + u_prescribed
            pos = r0 + u # postion at current epoch

            # Efficient site energies from fixed reference pair list
            site_E = site_energies_from_pair_list(pos, pair_i, pair_j)

            # energy of "atomistic-atoms"
            atomistic_energy = site_E[atomistic_indices_t].sum()

            # energy of clusters 
            cluster_energy = torch.sum(cac_weights_t * site_E[rep_indices_t])

            # total internal energy 
            internal_energy = atomistic_energy + cluster_energy

            # work done by external force (neumann BC) 
            external_work = torch.sum(f_ext * u) 
            # external_work = 0

            # loss-function
            loss_energy = internal_energy - external_work
            
            # true all-atom energy
            true_energy = site_E.sum() - external_work

            loss_energy.backward()
            adam_opt.step()

            if epoch >= MIN_EPOCHS_BEFORE_RECLUSTER and epoch % RECLUSTER_EVERY == 0:
                # Recompute current position after optimizer step
                with torch.no_grad():
                    u_raw_eval = model(r0_norm)
                    u_eval = mask * u_raw_eval + u_prescribed
                    pos_eval = r0 + u_eval
                    current_pos_np = pos_eval.detach().cpu().numpy()

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
                        jump_mad_factor=DEFGRAD_JUMP_MAD_FACTOR,
                        min_split_cluster_size=MIN_SPLIT_CLUSTER_SIZE,
                        min_cut_edges=MIN_DISCONTINUITY_EDGES,
                        max_iter=100,
                        tol=1e-10,
                    )
                )
                print(
                    "epoch", epoch,
                    "defgrad quadrature error",
                    recluster_info["rel_defgrad_quad_error"],
                    "full",
                    recluster_info["full_defgrad_value"],
                    "reduced",
                    recluster_info["reduced_defgrad_value"],
                    # "F_full",
                    # recluster_info['F_full']
                )

                if did_recluster:
                    split_info = recluster_info["split_info"]

                    print(
                        "quadrature triggered:",
                        recluster_info["quadrature_triggered"],
                    )
                    print(
                        "discontinuity triggered:",
                        recluster_info["discontinuity_triggered"],
                    )
                    print(
                        "clusters:",
                        split_info["num_clusters_before"],
                        "->",
                        split_info["num_clusters_after"],
                    )

                    for split in split_info["splits"]:
                        print("split:", split)

                    print(f"Re-clustering at epoch {epoch} using deformation-gradient weighted K-means")

                    cluster_idx = recluster_info["cluster_idx"]
                    centers = recluster_info["centers"]
                    rep_indices = recluster_info["rep_indices"]
                    cac_weights = recluster_info["cac_weights"]

                    rep_indices_t = torch.tensor(rep_indices,dtype=TORCH_LONG,device=TORCH_DEVICE)

                    cac_weights_t = torch.tensor(cac_weights,dtype=TORCH_FLOAT,device=TORCH_DEVICE,)

                    assert len(cluster_idx) == len(continuum_indices)
                    assert len(rep_indices) == len(cac_weights)
                    assert np.isclose(cac_weights.sum(), len(continuum_indices))
                    assert set(rep_indices).issubset(set(continuum_indices))

                    # Reset optimizer because the loss changed discontinuously
                    adam_opt = torch.optim.Adam(model.parameters(), lr=1e-4)

                    print("new rep_indices:", rep_indices)
                    print("new weights:", cac_weights)
                    print("sum weights:", cac_weights.sum())

            ################################3
            # log progress
            if epoch % LOG_EVERY == 0:
                with torch.no_grad():
                    u_raw_log = model(r0_norm)
                    u_log = mask * u_raw_log + u_prescribed
                    pos_log = r0 + u_log

                    site_E_log = site_energies_from_pair_list(pos_log,pair_i,pair_j,)

                    atomistic_energy_log = site_E_log[atomistic_indices_t].sum()

                    cluster_energy_log = torch.sum(cac_weights_t * site_E_log[rep_indices_t])

                    internal_energy_log = (atomistic_energy_log + cluster_energy_log)

                    external_work_log = torch.sum(f_ext * u_log)

                    loss_energy_log = (internal_energy_log - external_work_log)

                    true_internal_energy_log = site_E_log.sum()
                    true_total_potential_log = (true_internal_energy_log - external_work_log)

                current_pos_np = pos_log.detach().cpu().numpy()

                _, def_grad_norm, _ = local_deformation_gradient(reference_pos=atom_pos,current_pos=current_pos_np,neighs=neighs,)

                row = {
                    "epoch": int(epoch),
                    "loss_energy": loss_energy_log.cpu().item(),
                    "true_internal_energy": true_internal_energy_log.cpu().item(),
                    "true_total_potential": true_total_potential_log.cpu().item(),
                    "external_work": external_work_log.cpu().item(),
                    "pos": current_pos_np.tolist(),
                    "rep_indices": rep_indices.tolist(),
                    "cac_weights": cac_weights.tolist(),
                    "q_full": def_grad_norm.tolist(),
                }

                if not first_row:
                    jsonfile.write(",\n")
        
                jsonfile.write(json.dumps(row))
                first_row = False

                print(epoch, float(loss_energy_log.cpu().item()))
                jsonfile.flush()


        jsonfile.write("\n]\n")




def main():
    do_train_neuman_reclusterDEFGRAD()
    

if __name__ == "__main__":
    main()
