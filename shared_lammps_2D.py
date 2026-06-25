import numpy as np

# writes the lattice-configuration (initial positions of atoms)
# use atom_pos_defect for probems with defect
def write_lammps_data_2d(filename, pts,zlo=-10.0,zhi=10.0):
    """
        writes the initial positions of atoms to data-file
    """
    with open(filename, "w") as f:
        f.write("2D triangular LJ lattice\n\n")
        f.write(f"{len(pts)} atoms\n")
        f.write("1 atom types\n\n")

        xlo, ylo = pts.min(axis=0) - 5.0
        xhi, yhi = pts.max(axis=0) + 5.0

        f.write(f"{xlo:.16f} {xhi:.16f} xlo xhi\n")
        f.write(f"{ylo:.16f} {yhi:.16f} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi # large z-thickness to prevent interactoins between periodic layers\n\n") # the simulation box will have a large z-thickness and the atoms will be constrained in z-direction

        f.write("Masses\n\n")
        f.write("1 1.0\n\n")

        f.write("Atoms # atomic\n\n")
        for atom_id, (x, y) in enumerate(pts, start=1):
            f.write(f"{atom_id} 1 {x:.16f} {y:.16f} 0.0\n")

# helper function for converting python 0-based index of atoms into LAMMPS 1-based index 
def lammps_id_list(indices):
    """
    Convert 0-based Python atom indices to 1-based LAMMPS atom IDs.
    """
    return " ".join(str(int(i) + 1) for i in indices)

# helper function for Dirichlet Boundary Conditions
# These apply a very specific DBC!
def make_constraint_groups(atom_pos, tol=1e-12):
    """
    Return atom indices for:
        origin atom
        atoms on x-axis, excluding origin
        atoms on y-axis, excluding origin

    Python indices are 0-based.
    LAMMPS IDs will be index + 1.
    """
    x = atom_pos[:, 0]
    y = atom_pos[:, 1]

    origin = np.where((np.abs(x) < tol) & (np.abs(y) < tol))[0]

    xaxis = np.where((np.abs(y) < tol) & ~(np.abs(x) < tol))[0]
    yaxis = np.where((np.abs(x) < tol) & ~(np.abs(y) < tol))[0]

    if len(origin) != 1:
        raise RuntimeError(f"Expected exactly one origin atom, found {len(origin)}")

    return origin, xaxis, yaxis



def write_lammps_input_2d(
    filename,
    data_filename,
    atom_pos,
    spacing,
    cutoff_factor,
    dump_every,
    traj_filename,
    energy_log_filename,
    force_per_atom=None,
    right_band_width=None,
    right_indices=None,
    force_tol=1.0e-12,
    max_iter=10000,
    max_eval=100000,
):
    """
    Write LAMMPS input script for the 2D triangular LJ relaxation.

    Boundary conditions:
        Dirichlet:
            origin fixed in x and y
            x-axis atoms constrained in y
            y-axis atoms constrained in x

        Optional Neumann:
            constant force-per-atom applied to atoms in a right-side x-band

    Parameters
    ----------
    force_per_atom : None or array-like, shape (2,)
        If not None, applies [Fx, Fy] to each atom in the right boundary band.

    right_band_width : None or float
        Width of the right boundary band. If None and force_per_atom is not None,
        defaults to 0.6 * spacing.
    """

    origin, xaxis, yaxis = make_constraint_groups(atom_pos)

    rc = cutoff_factor * spacing

    # Optional Neumann BC setup
    use_force_bc = force_per_atom is not None
    fx,fy = 0.0,0.0
    if use_force_bc:
        if right_band_width is None:
            right_band_width = 0.6 * spacing

        force_per_atom = np.asarray(force_per_atom, dtype=float)

        if force_per_atom.shape != (2,):
            raise ValueError("force_per_atom must have shape (2,), e.g. np.array([Fx, Fy])")

        if len(right_indices) == 0:
            raise RuntimeError("No atoms selected in right boundary band.")

        fx = float(force_per_atom[0])
        fy = float(force_per_atom[1])

    with open(filename, "w") as f:
        f.write(f"""# ---------------------------------------------------------------
# LAMMPS input for 2D triangular LJ lattice
# ---------------------------------------------------------------

# Write LAMMPS screen/log output here.
# The thermo lines in this file contain energy every {dump_every} iterations.
log {energy_log_filename}

# LJ reduced units: epsilon = 1, sigma = 1, mass = 1
units lj

# True 2D simulation. LAMMPS requires z-periodicity for dimension 2.
dimension 2

# Simple atomic particles: id, type, x, y, z
atom_style atomic

# Finite in x and y, periodic in z as required for dimension 2
boundary f f p

# Read initial atom positions
read_data {data_filename}

# Physical cutoff:
# nearest neighbor distance = a
# second neighbor distance  = sqrt(3)*a
# third neighbor distance   = 2*a
# rc = {cutoff_factor}*a includes 1st and 2nd neighbors if cutoff_factor=1.9
pair_style lj/cut {rc:.16f}
pair_coeff 1 1 1.0 1.0 {rc:.16f}

# Neighbor list settings
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

# Keep system in the xy-plane
fix keep2d all enforce2d

# ---------------------------------------------------------------
# Boundary condition groups
# ---------------------------------------------------------------
""")

        f.write(f"group origin id {lammps_id_list(origin)}\n")
        f.write(f"group xaxis id {lammps_id_list(xaxis)}\n")
        f.write(f"group yaxis id {lammps_id_list(yaxis)}\n")

        if use_force_bc:
            f.write(f"group rightwall id {lammps_id_list(right_indices)}\n")

        f.write(f"""
# ---------------------------------------------------------------
# Dirichlet boundary constraints
# origin: fixed in x and y
# x-axis atoms: constrained in y
# y-axis atoms: constrained in x
# ---------------------------------------------------------------

fix fix_origin origin setforce 0.0 0.0 0.0
fix fix_xaxis xaxis setforce NULL 0.0 0.0
fix fix_yaxis yaxis setforce 0.0 NULL 0.0
""")

        if use_force_bc:
            f.write(f"""
# ---------------------------------------------------------------
# Neumann boundary condition
# Constant force-per-atom on right boundary band:
#     x >= xmax - right_band_width
#
# force_per_atom = [{fx:.16e}, {fy:.16e}]
# right_band_width = {right_band_width:.16e}
# number of loaded atoms = {len(right_indices)}
#
# This corresponds to the PINN term:
#     external_work = sum_i f_ext[i] dot u[i]
#     loss = internal_energy - external_work
# ---------------------------------------------------------------

fix load_right rightwall addforce {fx:.16e} {fy:.16e} 0.0
fix_modify load_right energy yes
""")

        f.write(f"""
# Print minimization progress every {dump_every} iterations.
# pe is total potential energy because thermo_modify norm no is used.
thermo {dump_every}
thermo_style custom step pe fnorm fmax
thermo_modify norm no

# Start minimization counter at 0
reset_timestep 0

# Write atom trajectory during minimization
dump traj all custom {dump_every} {traj_filename} id type x y z fx fy fz
dump_modify traj sort id

# Energy minimization
min_style cg
minimize 0.0 {force_tol:.16e} {max_iter} {max_eval}

# Stop trajectory dump
undump traj
""")

    print(f"Wrote LAMMPS input script: {filename}")
    print(f"Physical LJ cutoff rc = {rc:.16f}")
    print(f"Trajectory dump: {traj_filename}")
    print(f"Energy log: {energy_log_filename}")

    if use_force_bc:
        print("Applied Neumann force BC on right boundary band")
        print(f"right_band_width = {right_band_width}")
        print(f"number of loaded atoms = {len(right_indices)}")
        print(f"force_per_atom = [{fx}, {fy}]")
        print(f"total applied force = [{fx * len(right_indices)}, {fy * len(right_indices)}]")



# helper function for reading in LAMMPS output trajectory file
def read_lammps_trajectory_xy(filename):
    frames = []
    timesteps = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        timestep = int(lines[i + 1].strip())
        timesteps.append(timestep)

        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Unexpected dump format.")

        n_atoms = int(lines[i + 3].strip())

        # Skip box bounds:
        # ITEM: BOX BOUNDS ...
        # x bounds
        # y bounds
        # z bounds
        atom_header_index = i + 8

        if not lines[atom_header_index].startswith("ITEM: ATOMS"):
            raise RuntimeError("Could not find ITEM: ATOMS header.")

        atom_lines_start = atom_header_index + 1
        atom_lines_end = atom_lines_start + n_atoms

        ids = []
        coords = []

        for line in lines[atom_lines_start:atom_lines_end]:
            parts = line.split()
            atom_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])

            ids.append(atom_id)
            coords.append([x, y])

        ids = np.array(ids)
        coords = np.array(coords)

        order = np.argsort(ids)
        frames.append(coords[order])

        i = atom_lines_end

    return np.array(timesteps), np.array(frames)

# helper function for reading in LAMMPS energy-logs
import numpy as np


def read_lammps_thermo_energy(log_filename):
    """
    Read LAMMPS thermo output containing:

        Step PotEng c_pairpe Fnorm Fmax

    Returns
    -------
    steps : ndarray, shape (n,)
    potengs : ndarray, shape (n,)
        Total LAMMPS potential energy reported by PotEng.
    c_pairpes : ndarray, shape (n,)
        Pair-only potential energy from `compute pairpe all pe pair`.
    fnorms : ndarray, shape (n,)
    fmaxs : ndarray, shape (n,)
    """
    steps = []
    potengs = []
    c_pairpes = []
    fnorms = []
    fmaxs = []

    print("NOW READING ENERGY LOG", log_filename)

    expected_header = [
        "Step",
        "PotEng",
        "c_pairpe",
        "Fnorm",
        "Fmax",
    ]

    in_table = False

    with open(log_filename, "r") as f:
        for line in f:
            parts = line.split()

            # Detect the start of a thermo table.
            if parts[:5] == expected_header:
                in_table = True
                continue

            if not in_table:
                continue

            # Thermo data rows must have exactly five entries.
            if len(parts) != 5:
                # End the current table when LAMMPS prints text after it.
                if (
                    line.startswith("Loop time")
                    or line.startswith("Minimization stats")
                    or line.startswith("MPI task timing")
                ):
                    in_table = False

                continue

            try:
                step = int(parts[0])
                pe = float(parts[1])
                cpe = float(parts[2])
                fnorm = float(parts[3])
                fmax = float(parts[4])
            except ValueError:
                continue

            steps.append(step)
            potengs.append(pe)
            c_pairpes.append(cpe)
            fnorms.append(fnorm)
            fmaxs.append(fmax)

    if len(steps) == 0:
        raise RuntimeError(
            "No thermo rows were found. Expected header:\n"
            "Step PotEng c_pairpe Fnorm Fmax"
        )

    return (
        np.asarray(steps, dtype=int),
        np.asarray(potengs, dtype=float),
        np.asarray(c_pairpes, dtype=float),
        np.asarray(fnorms, dtype=float),
        np.asarray(fmaxs, dtype=float),
    )