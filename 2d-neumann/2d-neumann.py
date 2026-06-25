from shared_2D import *

# problem specification
NX, NY = 49, 50
LATTICE_SPACING = 2**(1/6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9 # for 2 nearest-neighbor interaction
PINN_HISTORY_JSON_FILENAME = "pinn_history/pinn-history-2d-neumann.json"
MAX_ITER = 10000
LOG_EVERY = 100
FORCE_PER_ATOM = np.array([0.01, 0.0])


# apply force-per-atom to atoms on the right boundary wall
# band-width determines how many atoms deep on the right-wall we will apply the force. we want to apply the force on two layers of atoms
def make_force_bc_right_boundary_band(atom_pos,force_per_atom,band_width,tol=1e-12):
    f_ext = np.zeros_like(atom_pos)

    x = atom_pos[:, 0]
    xmax = atom_pos[:, 0].max()

    right_boundary_band = x >= xmax - band_width - tol # atom-positions which have x = [xmax - bandwidth, xmax]

    right_boundary_indices = np.where(right_boundary_band)[0]

    f_ext[right_boundary_band, :] = force_per_atom

    return f_ext, right_boundary_band, right_boundary_indices


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

# mask for applying neumann boundary condition. dont apply DBC but you should hold some atoms fixed. 
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
weights_t = torch.tensor(weights, dtype=TORCH_FLOAT, device=TORCH_DEVICE)

# initialize the force-per-atom 
f_ext_np, right_boundary_band, right_boundary_indices = make_force_bc_right_boundary_band(atom_pos,force_per_atom=FORCE_PER_ATOM,band_width=0.6 * LATTICE_SPACING)
print("number of atoms under force:", np.sum(right_boundary_band))
f_ext = torch.tensor(f_ext_np, dtype=TORCH_FLOAT, device=TORCH_DEVICE)


import json
def do_train_DBC_NBC():
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
            cluster_energy = torch.sum(weights_t * site_E[rep_indices_t])

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

            ################################3
            # log progress
            if epoch % LOG_EVERY == 0:
                row = {
                    "epoch": int(epoch),
                    "loss_energy": loss_energy.detach().cpu().item(),
                    "true_energy": true_energy.detach().cpu().item(),
                    "pos": pos.detach().cpu().tolist()
                }
        
                if not first_row:
                    jsonfile.write(",\n")
        
                jsonfile.write(json.dumps(row))
                first_row = False

                print(epoch, float(loss_energy.detach()))
                jsonfile.flush()

        jsonfile.write("\n]\n")




# write out LAMMPS script and execute it as well. 
from shared_lammps_2D import *
import subprocess
LAMMPS_DIR          =   "./lammps_2d_input_output/"
LAMMPS_LATTICE_DATA =   LAMMPS_DIR + "lattice_2d.data"
LAMMPS_INPUT_SCRIPT =   LAMMPS_DIR + "in.relax_2d_neumann"
LAMMPS_TRAJ_FILE    =   LAMMPS_DIR + "relax_2d_neumann_traj.lammpstrj"
LAMMPS_ENERGY_LOG   =   LAMMPS_DIR + "relax_2d_neumann_energy.log"

def do_lammps():
    write_lammps_data_2d(filename=LAMMPS_LATTICE_DATA,pts=atom_pos,)

    write_lammps_input_2d(
        filename=LAMMPS_INPUT_SCRIPT,
        data_filename=LAMMPS_LATTICE_DATA,
        atom_pos=atom_pos,
        spacing=LATTICE_SPACING,
        cutoff_factor=CUTOFF_FACTOR,
        dump_every=LOG_EVERY,
        traj_filename=LAMMPS_TRAJ_FILE,
        energy_log_filename=LAMMPS_ENERGY_LOG,
        force_per_atom=FORCE_PER_ATOM,
        right_band_width=0.6 * LATTICE_SPACING,
        right_indices=right_boundary_indices,
        force_tol=1.0e-12,
    )

    # Define the LAMMPS command and input script arguments
    lammps_cmd = ["lmp", "-in", LAMMPS_INPUT_SCRIPT]

    # Launch the subprocess
    process = subprocess.run(lammps_cmd, capture_output=True, text=True)

    # Print the simulation output and errors
    print("Output:\n", process.stdout)
    if process.stderr:
        print("Errors:\n", process.stderr)




def main():
    do_train_DBC_NBC()
    do_lammps()
    

if __name__ == "__main__":
    main()