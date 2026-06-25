# kmeans weighted by deformation-gradient 
# reclustering triggered by variance of deformation-gradiet

from shared_2D import *

# problem specification
NX, NY = 49, 50
LATTICE_SPACING = 2**(1/6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9
MAX_ITER = 10000
LOG_EVERY = 100
PINN_HISTORY_JSON_FILENAME = "pinn-history-2d-basic/pinn-history-2d-recluster-defgrad.json"
MIN_EPOCHS_BEFORE_RECLUSTER = 1000
RECLUSTER_EVERY = 500
DEFGRAD_TRIGGER_TOL = 1.0e-3
DEFGRAD_JUMP_TOL = 5.0e-3
DEFGRAD_JUMP_MAD_FACTOR = 3.0
MIN_SPLIT_CLUSTER_SIZE = 10
MIN_DISCONTINUITY_EDGES = 3

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

# dont apply mask for DBC. instead use a mask for holding opposite side atoms fixed. 
# mask,u_prescribed = generate_dirichlet_mask(r0=r0, atom_pos=atom_pos)
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

import json 
def do_train_recluster_defgradvar():
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

            # total internal energy (used as loss function)
            loss_energy = atomistic_energy + cluster_energy    

            # true all-atom energy
            true_energy = site_E.sum()

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

                current_pos_np = pos_log.detach().cpu().numpy()
                _, def_grad_norm, _ = local_deformation_gradient(reference_pos=atom_pos,current_pos=current_pos_np,neighs=neighs,)
                
                row = {
                    "epoch": int(epoch),
                    "loss_energy": loss_energy.detach().cpu().item(),
                    "true_energy": true_energy.detach().cpu().item(),
                    # "pos": pos.detach().cpu().tolist(),
                    "pos": current_pos_np.tolist(),
                    "rep_indices":rep_indices.tolist(),
                    "q_full":def_grad_norm.tolist(),
                }
        
                if not first_row:
                    jsonfile.write(",\n")
        
                jsonfile.write(json.dumps(row))
                first_row = False

                print(epoch, float(loss_energy.detach()))
                jsonfile.flush()


        jsonfile.write("\n]\n")




def main():
    do_train_recluster_defgradvar()
    

if __name__ == "__main__":
    main()
