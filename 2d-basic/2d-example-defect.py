from shared_2D import *

# problem specification
NX, NY = 49, 50
LATTICE_SPACING = 2**(1/6)
BORDER_LAYERS = 3
CUTOFF_FACTOR = 1.9
DEFECT_LAYERS= 3
PINN_HISTORY_JSON_FILENAME = "pinn_history/pinn-history-2d-defect.json"
MAX_ITER = 10000
LOG_EVERY = 100

atom_pos_full = make_triangular_lattice(nx=NX, ny=NY, spacing=LATTICE_SPACING)

atom_pos_defect, defect_pos, defect_old_idx = remove_central_atom(atom_pos_full)

print("Original atoms:", len(atom_pos_full))
print("After point defect:", len(atom_pos_defect))
print("Removed old index:", defect_old_idx)
print("Defect position:", defect_pos)

# 3 layers of atoms bordering the square are flagged as atomistic-region
atomistic_indices,continuum_indices = split_atomistic_continuum_defect(atom_pos_defect, defect_pos, NX,NY,LATTICE_SPACING,BORDER_LAYERS,DEFECT_LAYERS)
continuum_atom_pos = atom_pos_defect[continuum_indices]
atomistic_atom_pos = atom_pos_defect[atomistic_indices]

# manually pick 5 rep_atoms among the continuum atoms 
initial_rep_indices, rep_targets = pick_manual_rep_atoms_defect(atom_pos_defect,continuum_indices,0.10)

# assign continuum-atoms to their clusters, and equilibrate over 100 iterations of kmeans 
cluster_idx, centers, rep_indices, weights, history = kmeans_unweighted(atom_pos_defect,continuum_indices,initial_rep_indices)

print("number of atoms \t\t", len(atom_pos_defect))
print("atomistic atoms \t\t", len(atomistic_indices))
print("continuum atoms \t\t", len(continuum_indices))
print("initial rep atom indices \t", initial_rep_indices)
print("final rep atom indices \t\t", rep_indices)
print("weights \t\t\t", weights)
print("sum of weights \t\t\t", weights.sum())

# normalized inputs
r0,r0_norm = generate_normalized_input(atom_pos=atom_pos_defect)

mask,u_prescribed = generate_dirichlet_mask(r0,atom_pos_defect)


### Set up the neighbrlist once:
pair_i_np, pair_j_np = build_reference_pair_list(atom_pos_defect, cutoff = CUTOFF_FACTOR*LATTICE_SPACING)
pair_i = torch.tensor(pair_i_np, dtype=TORCH_LONG, device=TORCH_DEVICE)
pair_j = torch.tensor(pair_j_np, dtype=TORCH_LONG, device=TORCH_DEVICE)


# initialize the PINN 
model = PINN2D().to(device=TORCH_DEVICE, dtype=TORCH_FLOAT)
adam_opt = torch.optim.Adam(model.parameters(), lr=1e-4) # optimizer adjusts the weights of the NN


atomistic_indices_t = torch.tensor(atomistic_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
rep_indices_t = torch.tensor(rep_indices, dtype=TORCH_LONG, device=TORCH_DEVICE)
weights_t = torch.tensor(weights, dtype=TORCH_FLOAT, device=TORCH_DEVICE)


import json
def do_train_defect_DBC_no_external_force():
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
        
            # approximate QC/PINN energy
            loss_atomistic = site_E[atomistic_indices_t].sum()
            loss_reps = torch.sum(weights_t * site_E[rep_indices_t])
            loss = loss_atomistic + loss_reps
            
            # true all-atom energy
            true_energy = site_E.sum()

            loss.backward()
            adam_opt.step()

            ################################
            # log progress
            if epoch % LOG_EVERY == 0:
                row = {
                    "epoch": int(epoch),
                    "loss_energy": loss.detach().cpu().item(),
                    "true_energy": true_energy.detach().cpu().item(),
                    "pos": pos.detach().cpu().tolist()
                }

                if not first_row:
                    jsonfile.write(",\n")

                jsonfile.write(json.dumps(row))
                first_row = False

                print(epoch, float(loss.detach()))
                jsonfile.flush()

        jsonfile.write("\n]\n")




# write out LAMMPS script and execute it as well. 
from shared_lammps_2D import *
import subprocess
LAMMPS_DIR          =   "./lammps_2d_input_output/"
LAMMPS_LATTICE_DATA =   LAMMPS_DIR + "lattice_2d_defect.data"
LAMMPS_INPUT_SCRIPT =   LAMMPS_DIR + "in.relax_2d_defect"
LAMMPS_TRAJ_FILE    =   LAMMPS_DIR + "relax_2d_defect_traj.lammpstrj"
LAMMPS_ENERGY_LOG   =   LAMMPS_DIR + "relax_2d_defect_energy.log"

def do_lammps():
    write_lammps_data_2d(filename=LAMMPS_LATTICE_DATA,pts=atom_pos_defect,)

    write_lammps_input_2d(
        filename=LAMMPS_INPUT_SCRIPT,
        data_filename=LAMMPS_LATTICE_DATA,
        atom_pos=atom_pos_defect,
        spacing=LATTICE_SPACING,
        cutoff_factor=1.9,
        dump_every=LOG_EVERY,
        max_iter=MAX_ITER,
        traj_filename=LAMMPS_TRAJ_FILE,
        energy_log_filename=LAMMPS_ENERGY_LOG,
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
    do_train_defect_DBC_no_external_force()
    do_lammps()
    

if __name__ == "__main__":
    main()