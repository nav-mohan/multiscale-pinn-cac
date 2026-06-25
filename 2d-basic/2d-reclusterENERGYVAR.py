from shared_2D import *

# problem specification
NX, NY = 49, 50
LATTICE_SPACING = 2**(1/6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9
MAX_ITER = 10000
LOG_EVERY = 100
PINN_HISTORY_JSON_FILENAME = "pinn-history-2d-basic/pinn-history-2d-recluster-energyvar.json"

atom_pos = make_triangular_lattice(NX,NY,LATTICE_SPACING)

# 3 layers of atoms bordering the square are flagged as atomistic-region
atomistic_indices,continuum_indices = split_atomistic_continuum(NX,NY,BORDER_LAYERS)
continuum_atom_pos = atom_pos[continuum_indices]
atomistic_atom_pos = atom_pos[atomistic_indices]

# manually pick 5 rep_atoms among the continuum atoms 
initial_rep_indices, rep_targets = pick_manual_rep_atoms(atom_pos,continuum_indices,0.1)

# assign continuum-atoms to their clusters, and equilibrate over 100 iterations of kmeans 
cluster_idx, centers, rep_indices, weights, history = kmeans_unweighted(atom_pos,continuum_indices,initial_rep_indices)

print("number of atoms \t\t", len(atom_pos))
print("atomistic atoms \t\t", len(atomistic_indices))
print("continuum atoms \t\t", len(continuum_indices))
print("initial rep atom indices \t", initial_rep_indices)
print("final rep atom indices \t\t", rep_indices)
print("weights \t\t\t", weights)
print("sum of weights \t\t\t", weights.sum())

# normalized inputs
r0,r0_norm = generate_normalized_input(atom_pos=atom_pos)

# mask for applying dirichlet boundary condition. this is a very specific Dirichlet BC
mask,u_prescribed = generate_dirichlet_mask(r0=r0, atom_pos=atom_pos)

### Set up the neighbrlist once:
pair_i_np, pair_j_np = build_reference_pair_list(atom_pos, cutoff = CUTOFF_FACTOR*LATTICE_SPACING)
pair_i = torch.tensor(pair_i_np, dtype=TORCH_LONG, device=TORCH_DEVICE)
pair_j = torch.tensor(pair_j_np, dtype=TORCH_LONG, device=TORCH_DEVICE)

# initialize the PINN 
model = PINN2D().to(device=TORCH_DEVICE, dtype=TORCH_FLOAT)
adam_opt = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer adjusts the weights of the NN


atomistic_indices_t = torch.tensor(atomistic_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
rep_indices_t = torch.tensor(rep_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
weights_t = torch.tensor(weights, dtype=TORCH_FLOAT, device=TORCH_DEVICE)


def cluster_rep_energy_mismatch_indicator(
    site_energies_np,
    continuum_indices,
    cluster_idx,
    rep_indices,
):
    """
    Indicator based on mismatch between each atom in a cluster
    and that cluster's representative atom.
    """
    continuum_E = site_energies_np[continuum_indices]

    n_rep = len(rep_indices)
    indicators = np.zeros(n_rep, dtype=np.float64)

    for k in range(n_rep):
        E_cluster = continuum_E[cluster_idx == k]

        if len(E_cluster) == 0:
            indicators[k] = 0.0
            continue

        E_rep = site_energies_np[rep_indices[k]]

        mismatch = E_cluster - E_rep

        indicators[k] = np.sqrt(np.mean(mismatch**2)) / (abs(E_rep) + 1e-12)

    return indicators


def split_bad_clusters_by_energy(current_pos_np,continuum_indices,cluster_idx,rep_indices,weights,bad_clusters,):
    """
    Split each bad cluster into two clusters using a simple geometric split.

    This increases the number of representative atoms.
    """
    new_rep_indices = []
    new_cluster_labels = np.full_like(cluster_idx, fill_value=-1)
    new_label = 0

    continuum_pos = current_pos_np[continuum_indices]

    for k in range(len(rep_indices)):
        local_members = np.where(cluster_idx == k)[0]

        if len(local_members) == 0:
            continue

        # Keep good clusters as-is
        if k not in bad_clusters:
            new_rep_indices.append(rep_indices[k])
            new_cluster_labels[local_members] = new_label
            new_label += 1
            continue

        # Split bad cluster into 2 subclusters
        member_pos = continuum_pos[local_members]

        # Pick two farthest-ish seeds: first seed = old rep,
        # second seed = farthest member from old rep.
        old_rep_pos = current_pos_np[rep_indices[k]]
        d = np.linalg.norm(member_pos - old_rep_pos[None, :], axis=1)
        seed1 = old_rep_pos
        seed2 = member_pos[np.argmax(d)]

        centers = np.vstack([seed1, seed2])

        # A few local K-means iterations
        for _ in range(20):
            dist = np.linalg.norm(member_pos[:, None, :] - centers[None, :, :],axis=2)
            sublabels = np.argmin(dist, axis=1)

            for s in range(2):
                if np.any(sublabels == s):
                    centers[s] = member_pos[sublabels == s].mean(axis=0)

        # Assign final sublabels
        dist = np.linalg.norm(member_pos[:, None, :] - centers[None, :, :],axis=2)
        sublabels = np.argmin(dist, axis=1)

        for s in range(2):
            sub_members = local_members[sublabels == s]

            if len(sub_members) == 0:
                continue

            sub_pos = continuum_pos[sub_members]
            dcenter = np.linalg.norm(sub_pos - centers[s][None, :], axis=1)
            nearest_local = sub_members[np.argmin(dcenter)]
            rep_global_idx = continuum_indices[nearest_local]

            new_rep_indices.append(rep_global_idx)
            new_cluster_labels[sub_members] = new_label
            new_label += 1

    new_rep_indices = np.array(new_rep_indices, dtype=int)

    new_weights = np.array(
        [np.sum(new_cluster_labels == k) for k in range(len(new_rep_indices))],
        dtype=np.float64,
    )

    return new_cluster_labels, new_rep_indices, new_weights



import json
def do_train_DBC_no_external_force_recluster():
    global cluster_idx
    global rep_indices
    global weights
    global rep_indices_t
    global weights_t
    global adam_opt

    max_iter = 10000
    print_every = 100
    mismatch_threshold = 3e-6   # mismatch-norm =%
    min_epoch_before_recluster = 1000
    recluster_every = 500
    with open(PINN_HISTORY_JSON_FILENAME, "w") as jsonfile:
        jsonfile.write("[\n")
        first_row = True

        for epoch in range(max_iter):
            adam_opt.zero_grad()

            u_raw = model(r0_norm)
            u = mask * u_raw + u_prescribed
            pos = r0 + u # postion at current epoch

            # Efficient site energies from fixed reference pair list
            site_E = site_energies_from_pair_list(pos, pair_i, pair_j)

            # energy of "atomistic-atoms"
            atomistic_energy = site_E[atomistic_indices_t].sum()
            
            # energy of clusters
            cluster_energy = torch.sum(weights_t * site_E[rep_indices_t])

            # total internal energy (used as loss function)
            loss_energy = atomistic_energy + cluster_energy    

            # true all-atom energy
            true_energy = site_E.sum()

            loss_energy.backward()
            adam_opt.step()

            ################################
            # recluster required?
            if epoch > min_epoch_before_recluster and epoch % recluster_every == 0:
                with torch.no_grad():
                    # evaluate true atomic-energies for current epoch's configuration
                    site_E = site_energies_from_pair_list(pos, pair_i, pair_j) 
                    site_E_np = site_E.detach().cpu().numpy()
            
                # evaluate true atomic-energies for all continuum atoms
                continuum_E = site_E_np[continuum_indices]


                eta = cluster_rep_energy_mismatch_indicator(
                    site_energies_np=site_E_np,
                    continuum_indices=continuum_indices,
                    cluster_idx=cluster_idx,
                    rep_indices=rep_indices,
                )
                # print(f"eta = {eta}")
                bad_clusters = np.where(eta > mismatch_threshold)[0]
                
                if len(bad_clusters) > 0:
                    print(f"re-clustering triggered at epoch {epoch}")
                    print(f"max eta = {np.max(eta):.6e}")
                    current_pos_np = pos.detach().cpu().numpy()
                
                    cluster_idx, rep_indices, weights = split_bad_clusters_by_energy(
                        current_pos_np=current_pos_np,
                        continuum_indices=continuum_indices,
                        cluster_idx=cluster_idx,
                        rep_indices=rep_indices,
                        weights=weights,
                        bad_clusters=set(bad_clusters),
                    )
                
                    # Update Torch tensors used in the loss
                    rep_indices_t = torch.tensor(rep_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
                    weights_t = torch.tensor(weights, dtype=TORCH_FLOAT, device=TORCH_DEVICE)
                    print("new number of representative atoms:", len(rep_indices))
                    # print("updated rep_indices:", rep_indices)
                    # print("updated weights:", weights)
                    
                    # reset the optimizer. increasing the number of rep-atoms will lead to
                    # a discontinuous jump in gradients between this epoch and next epoch
                    adam_opt = torch.optim.Adam(model.parameters(), lr=1e-4)

            ################################3
            # log progress
            if epoch % print_every == 0:
                row = {
                    "epoch": int(epoch),
                    "loss_energy": loss_energy.detach().cpu().item(),
                    "true_energy": true_energy.detach().cpu().item(),
                    "pos": pos.detach().cpu().tolist(),
                    "rep_indices":rep_indices.tolist()
                }
        
                if not first_row:
                    jsonfile.write(",\n")
        
                jsonfile.write(json.dumps(row))
                first_row = False

                print(epoch, float(loss_energy.detach()))
                jsonfile.flush()


        jsonfile.write("\n]\n")



##################### DONT HAVE TO DO LAMMPS ####################
##################### JUST REUSE 2D-EXAMPLE's ####################


def main():
    do_train_DBC_no_external_force_recluster()
    

if __name__ == "__main__":
    main()