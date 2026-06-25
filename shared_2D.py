"""
shared code for all 2D problems. 
    
    make_triangular_lattice - generates a hexagonal lattice 
    kmeans_unweighted - geometric kmeans clustering. without physics. 

"""

import torch
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt
import json 

TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_FLOAT = torch.float64
TORCH_LONG = torch.long


# make a hexagonal lattice. return an array of shape (2450,2). 
def make_triangular_lattice(nx=49, ny=50, spacing=2**(1/6)):
    """
        atom_k has index k = j*nx + i 
        where i = column-index, 0 <= i < nx
        where j = row-index, 0 <= j < ny
    """
    atom_pos = []

    for j in range(ny):
        for i in range(nx):
            x = spacing * (i + 0.5 * (j % 2))
            y = spacing * (np.sqrt(3) / 2) * j
            atom_pos.append([x, y])

    return np.array(atom_pos, dtype=np.float64)


# assign the clusters and equilibrate a little 
# use atom_pos_defect when applying to with-defect problems
def kmeans_unweighted(atom_pos, continuum_indices, initial_rep_indices, max_iter = 100, tol = 1e-10):
    """
        when running this on defect-system, remember to atom_pos_defect not atom_pos
    """
    continuum_atoms = atom_pos[continuum_indices]
    centers = atom_pos[initial_rep_indices].copy()

    n_rep = len(initial_rep_indices)
    history = []

    # assignment step 
    for it in range(max_iter):
        # find distance of all continuum atoms to all rep_atoms 
        dist_matrix = np.linalg.norm(continuum_atoms[:,None,:] - centers[None, :, :], axis=2)

        cluster_idx = np.argmin(dist_matrix,axis=1) 
        # cluster_idx is an array of len continuum_atoms
        # cluster_idx[i]=k means continuum_atoms[i] belongs to rep_atom_k's cluster

        J = np.sum((continuum_atoms - centers[cluster_idx])**2)
        history.append(J)

        # update step 
        new_centers = centers.copy()

        for k in range(n_rep):
            members = continuum_atoms[cluster_idx==k] # all the continuum_atoms that are members of cluster-k

            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)

        # check convergence
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol: break 

    # do one final evaluation
    dist_matrix = np.linalg.norm(continuum_atoms[:,None,:] - centers[None, :, :], axis=2)
    cluster_idx = np.argmin(dist_matrix,axis=1)

    # assign weigths (count the number of atoms in each cluster)
    weights = np.array([np.sum(cluster_idx == k) for k in range(n_rep)],dtype=np.float64)

    # after performing equilibration of cluster-centers, convert cluster-centers to atom_indices of rep_atoms
    final_rep_idx = []
    for k in range(n_rep):
        
        local_members = np.where(cluster_idx==k)[0]
        
        if len(local_members) == 0:
            final_rep_idx.append(initial_rep_indices[k])
            continue

        cluster_atoms = continuum_atoms[local_members]
        d = np.linalg.norm(cluster_atoms - centers[k][None,:],axis=1)
        nearest_local_member = local_members[np.argmin(d)]
        final_rep_idx.append(continuum_indices[nearest_local_member])

    final_rep_idx = np.array(final_rep_idx,dtype=int)
        
    return cluster_idx, centers, final_rep_idx, weights, history


# use atom_pos_defect when applying to with-defect problems
def kmeans_weighted(atom_pos,continuum_indices,initial_rep_indices,
                    importance_weights,
                    max_iter=100,tol=1e-10,min_importance=1e-8,):
    """
    Physics-weighted K-means on continuum atoms.

    The K-means objective is:

        J = sum_i q_i ||x_i - c_{cluster(i)}||^2

    where q_i is an importance weight for continuum atom i.

    Parameters
    ----------
    atom_pos : np.ndarray, shape (N, 2)
        Current atom positions. Usually current deformed positions.

    continuum_indices : np.ndarray, shape (N_continuum,)
        Global atom indices of continuum atoms.

    initial_rep_indices : np.ndarray, shape (n_rep,)
        Global atom indices used to initialize cluster centers.

    importance_weights : np.ndarray, shape (N_continuum,)
        Physics weights q_i for continuum atoms only.
        Larger q_i means atom i pulls cluster centers more strongly.

    Returns
    -------
    cluster_idx : np.ndarray, shape (N_continuum,)
        cluster_idx[m] = k means continuum_indices[m] belongs to cluster k.

    centers : np.ndarray, shape (n_rep, 2)
        Final weighted cluster centers.

    final_rep_idx : np.ndarray, shape (n_rep,)
        Actual atom indices nearest to each final center.

    cac_weights : np.ndarray, shape (n_rep,)
        CAC quadrature weights. These are atom counts per cluster.

    history : list[float]
        Weighted K-means objective history.
    """

    continuum_atoms = atom_pos[continuum_indices]
    centers = atom_pos[initial_rep_indices].copy()

    q = np.asarray(importance_weights, dtype=np.float64).copy()

    if q.shape[0] != len(continuum_indices):
        raise ValueError(
            "importance_weights must have length len(continuum_indices). "
            f"Got {q.shape[0]} but expected {len(continuum_indices)}."
        )

    # Keep weights positive and normalized.
    q = np.maximum(q, min_importance)
    q = q / (np.mean(q) + 1e-12)

    n_rep = len(initial_rep_indices)
    history = []

    for it in range(max_iter):
        dist2 = np.sum(
            (continuum_atoms[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )

        cluster_idx = np.argmin(dist2, axis=1)

        # use weights when computing J 
        J = np.sum(q * dist2[np.arange(len(continuum_atoms)), cluster_idx])
        history.append(J)

        new_centers = centers.copy()

        for k in range(n_rep):
            members = cluster_idx == k

            if np.any(members):
                qk = q[members]
                xk = continuum_atoms[members]

                # use weights when finding new centers
                new_centers[k] = np.sum(qk[:, None] * xk, axis=0) / (np.sum(qk) + 1e-12) 

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if shift < tol:
            break

    # Final assignment after convergence
    dist2 = np.sum(
        (continuum_atoms[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )

    cluster_idx = np.argmin(dist2, axis=1)

    # IMPORTANT:
    # CAC weights are still cluster atom counts, not sums of q.
    cac_weights = np.array(
        [np.sum(cluster_idx == k) for k in range(n_rep)],
        dtype=np.float64,
    )

    final_rep_idx = []

    for k in range(n_rep):
        local_members = np.where(cluster_idx == k)[0]

        if len(local_members) == 0:
            final_rep_idx.append(initial_rep_indices[k])
            continue

        cluster_atoms = continuum_atoms[local_members]
        d = np.linalg.norm(cluster_atoms - centers[k][None, :], axis=1)
        nearest_local_member = local_members[np.argmin(d)]

        final_rep_idx.append(continuum_indices[nearest_local_member])

    final_rep_idx = np.array(final_rep_idx, dtype=int)

    return cluster_idx, centers, final_rep_idx, cac_weights, history


# plot continuum-clusters, atomistic-atoms, cluster-centers, rep-atoms
# use atom_pos_defect when applying to with-defect problems
def plot_partition(atom_pos, atomistic_indices, continuum_indices, cluster_idx, rep_indices, centers,weights,figure_size=(8,7),plot_title="2D K-means partition"):
    plt.figure(figsize=figure_size)

    cont_atom_pos = atom_pos[continuum_indices]
    atom_atom_pos = atom_pos[atomistic_indices]
    rep_atom_pos = atom_pos[rep_indices]

    # plot continuum region atoms 
    plt.scatter(cont_atom_pos[:, 0],cont_atom_pos[:, 1],c=cluster_idx,s=8,alpha=0.6,label='continuum region')

    # plot atomistic_regions atoms 
    plt.scatter(atom_atom_pos[:, 0],atom_atom_pos[:, 1],c="black",s=10,label="atomistic region",)

    # plot cluster centers 
    plt.scatter(centers[:, 0],centers[:, 1],c="red",marker="x",s=50,linewidths=1,label="cluster centers",)

    # highlight the atoms that have been assigned as rep_atoms
    plt.scatter(rep_atom_pos[:, 0],rep_atom_pos[:, 1],facecolors="none",edgecolors="red",s=180,linewidths=2,label="rep_atoms",)

    # textbox for weights
    for k, idx in enumerate(rep_indices):
        x, y = atom_pos[idx]
        plt.text(x + 0.2, y + 0.2, f"w={int(weights[k])}", fontsize=9)

    plt.axis("equal")
    plt.legend()
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()





"""
#####################################
PINN2D, SWISH-ACTIVATION, GLOROT-WEIGHTS
#####################################
"""

class SwishActivation(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
        
class PINN2D(nn.Module):

    def GlorotInitWeights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
    def __init__(self,width=23,depth=4): # 4 hidden layers. 23 neurons per hidden layer
        super().__init__()
    
        # input layer. 2 input features (x,y of atom coords) and 23 output features with Swish activation
        # each atom coordinate is mapped to 23 numbers
        layers = [nn.Linear(2,width),SwishActivation()] 

        # we've already created the first hidden layer above. now add 3 more layers
        # each of these layers have 23 inputs and 23 output features with Swish activation
        for _ in range(depth-1):
            layers += [nn.Linear(width,width),SwishActivation()]

        # add final output layer. 23 inputs and 2 outputs (ux,uy of displacement)
        layers += [nn.Linear(width,2)]

        # make a neural-network of all these layers
        self.net = nn.Sequential(*layers)

        self.apply(self.GlorotInitWeights)

    # forward is a method that gets automatically called when you call the model u_raw = model(x)
    def forward(self,x):
        return self.net(x) # apply the neural network on the input coordination (x,y) of each atom
        

"""
    LENNARD-JONES INTERACTION
    SITE-ENERGY CALUCATION HELPERS 
    NEIGHBORLIST BUILDER HELPERS
"""

def LennardJones6(r, epsilon=1.0, sigma=1.0):
    sigma_r_6 = (sigma / r) ** 6
    return 4 * epsilon * (sigma_r_6**2 - sigma_r_6)

# build a pair of "parallel-arrays" where pair_i[k] and pair_j[k] are neighbors
# use atom_pos_defect when applying to with-defect problems
def build_reference_pair_list(atom_pos, cutoff):
    """
        we are reusing the same neighborlist, so we build the neighborlist-mapping once and computing atomic-energies is much more efficient than N^2
        pair_i and pair_j are two "parallel" arrays of interacing atoms. 
        pair_i[i] and pair_j[i] are interacting 
        Build unique neighbor pairs (i, j), i < j, from the reference configuration.

        atom_pos: NumPy array, shape (N, 2)
        cutoff: physical cutoff radius

        Returns:
            pair_i, pair_j as NumPy arrays
    """
    N = len(atom_pos)

    pair_i = []
    pair_j = []

    for i in range(N):
        # distance between atom_i and atoms_j, for all j > i
        dist = np.linalg.norm(atom_pos[i + 1:] - atom_pos[i], axis=1)

        local_js = np.where(dist < cutoff)[0] + (i + 1)

        for j in local_js:
            pair_i.append(i)
            pair_j.append(j)

    return np.array(pair_i, dtype=np.int64), np.array(pair_j, dtype=np.int64)



# a helper method to compute the energies of all atoms using the above parallel lists
def site_energies_from_pair_list(pos, pair_i, pair_j):
    """
        Compute site energy E_i for every atom using a fixed pair list.

        pos: torch tensor, shape (N, 2)
        pair_i, pair_j: torch long tensors, shape (N_pairs,)

        Returns:
            site_energy: torch tensor, shape (N,)
    """
    rij = pos[pair_j] - pos[pair_i]
    r = torch.linalg.norm(rij, dim=1)

    pair_energy = LennardJones6(r)

    site_energy = torch.zeros(pos.shape[0],dtype=pos.dtype,device=pos.device)

    # Each pair energy is split half-half between the two atoms.
    half_pair_energy = 0.5 * pair_energy

    site_energy.index_add_(0, pair_i, half_pair_energy)
    site_energy.index_add_(0, pair_j, half_pair_energy)

    return site_energy



'''
### how to use the above parallel-arrays: 
#### Set up the neighbrlist once (`pair_i`, `pair_j` are the parallel-arrays):
    pair_i_np, pair_j_np = build_reference_pair_list(atom_pos, cutoff = 1.9*spacing)
    pair_i = torch.tensor(pair_i_np, dtype=TORCH_LONG, device=device)
    pair_j = torch.tensor(pair_j_np, dtype=TORCH_LONG, device=device)

#### then during training, every 500 epochs, compute the "true" energy of all atoms
    site_E = site_energies_from_pair_list(r0, pair_i, pair_j)
    true_total_energy = site_E.sum()

#### and then compute the within-cluster standard-deviation
    for k in range(len(rep_indices)):
        print(site_E[np.where(cluster_idx == k)].std())
'''

# generate normalized inputs of atom coordinates
# use atom_pos_defect for problems with defect
def generate_normalized_input(atom_pos):
    """
        normalizes by usual (x - mean_x)/std_x
        returns torch tensors r0 and r0_norm 
    """
    r0 = torch.tensor(atom_pos, dtype=TORCH_FLOAT, device=TORCH_DEVICE) # convert atomic-positions to torch-tensro

    # normalize the inputs
    mu = r0.mean(dim=0, keepdim=True)
    std = r0.std(dim=0, keepdim=True)
    r0_norm = (r0 - mu) / std

    return r0,r0_norm

# generate the mask that applies dirichlet boundary conditions on atoms
# use atom_pos_defect for problems with defect 
def generate_dirichlet_mask(r0,atom_pos):
    """
        this generates a VERY SPECIFIC dirichley boundary condition.

        origin atom:
            ux = 0, uy = 0

        atoms on x-axis:
            uy = 0, ux free

        atoms on y-axis:
            ux = 0, uy free

        u_prescribed: 
    """
    mask = torch.ones_like(r0, device=TORCH_DEVICE)
    u_prescribed = torch.zeros_like(r0, device=TORCH_DEVICE)

    tol = 1e-12
    on_x_axis = np.abs(atom_pos[:, 1]) < tol # if y == 0 (the first 49 atoms )
    on_y_axis = np.abs(atom_pos[:, 0]) < tol # if x == 0 (every 50th atom) 
    origin = on_x_axis & on_y_axis

    origin = torch.tensor(origin,device=TORCH_DEVICE,dtype=torch.bool)
    on_x_axis = torch.tensor(on_x_axis,device=TORCH_DEVICE,dtype=torch.bool)
    on_y_axis = torch.tensor(on_y_axis,device=TORCH_DEVICE,dtype=torch.bool)

    mask[on_x_axis, 1] = 0.0 # y-value of mask[i,j] set to 0
    mask[on_y_axis, 0] = 0.0 # x-value of mask[i,j] set to 0
    mask[origin, :] = 0.0 # x and y value of mask[i,j] set to 0

    return mask,u_prescribed


def generate_neumann_support_mask(r0, atom_pos, tol=1e-8):
    mask = torch.ones_like(r0)
    u_prescribed = torch.zeros_like(r0)

    xmin = atom_pos[:, 0].min()
    left_indices = np.where(
        np.abs(atom_pos[:, 0] - xmin) < tol
    )[0]

    # Roller support: prevent horizontal motion on the left.
    mask[left_indices, 0] = 0.0

    # Prevent rigid vertical translation using one left atom.
    anchor_idx = left_indices[
        np.argmin(atom_pos[left_indices, 1])
    ]
    mask[anchor_idx, 1] = 0.0

    return mask, u_prescribed


#######################################################################################
############################# SHARED CODE FOR ERROR ANALYSIS ##########################
#######################################################################################

# the first-neighbor-list is evaluated on the initial configuration. 
# this can be used for defect_pos and atom_pos
def build_first_neighbor_list(reference_pos, spacing=2**(1/6), tol=1e-6):
    """
    Build first-neighbor list. safe for a defective lattice also.

    This uses geometry only, so it works even if atoms have been removed.
    """
    neighs = []

    for i in range(len(reference_pos)):
        dist = np.linalg.norm(reference_pos - reference_pos[i], axis=1)

        nbrs = np.where(
            (dist > 1e-12) &
            (np.abs(dist - spacing) < tol)
        )[0]

        neighs.append(nbrs)

    return neighs


def bond_length_vector_from_neighs(pos, neighs):
    b = np.zeros(len(pos), dtype=np.float64)

    for i, nbrs in enumerate(neighs):
        if len(nbrs) == 0:
            b[i] = 0.0
            continue

        distances = np.linalg.norm(pos[nbrs] - pos[i], axis=1)
        b[i] = 0.5 * np.sum(distances)

    return b


def relative_bond_length_error(pos0, pos_lammps, pos_pinn, spacing=2**(1/6), tol=1e-6):
    neighs = build_first_neighbor_list(pos0, spacing=spacing, tol=tol)

    b_lammps = bond_length_vector_from_neighs(pos_lammps, neighs)
    b_pinn = bond_length_vector_from_neighs(pos_pinn, neighs)

    err = np.linalg.norm(b_lammps - b_pinn) / np.linalg.norm(b_lammps)

    return err, b_lammps, b_pinn



###########################################################
########## CODE FOR 2D LATTICE WITHOUT DEFECT #############
###########################################################

""" 
    shared code for 2D problems WITHOUT DEFECTS. 
    Only for problems WITHOUT DEFECTS!!
    
    split_atomistic_continuum_without_defect 
        outermost 3 layers of atoms are atomistic. rest are all continuum 
    
    pick_manual_rep_atoms_without_defect
        5 rep-atoms are distributed uniformly
    
    THIS IS ONLY FOR PROBLEMS WIHOUT DEFECTS
        problems with defects will have different split_atomistic_continuum
        problems with defects iwll have different pick_manual_rep_atoms
     
"""

def split_atomistic_continuum(nx=49, ny=50, border_layers=3):
    """
        splits the lattice into continuum and atomistic atoms
        outermost 3 layers of atoms are atomistic. rest are all continuum 
        uses flattened index:index = j * nx + i
        returns atomistic_indices, continuum_indices 
    """
    atomistic_indices = []
    continuum_indices = []

    for idx in range(nx * ny):
        j = idx // nx
        i = idx % nx

        is_border_atom = (
            i < border_layers
            or i >= nx - border_layers
            or j < border_layers
            or j >= ny - border_layers
        )

        if is_border_atom:
            atomistic_indices.append(idx)
        else:
            continuum_indices.append(idx)

    return (
        np.array(atomistic_indices, dtype=int),
        np.array(continuum_indices, dtype=int),
    )


# place 5 rep-atoms on the square. 
# one in the center. and the other 4 evenly spaced out lower-left, lower-right, upper-left, upper-right

def pick_manual_rep_atoms(atom_pos, continuum_indices,frac=0.25):
    cont_atom_pos = atom_pos[continuum_indices]

    xmin, ymin = cont_atom_pos.min(axis=0)
    xmax, ymax = cont_atom_pos.max(axis=0)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    # One center + four evenly distributed interior targets
    targets = np.array([
        [cx, cy],                                      # center
        [xmin + frac * (xmax - xmin), ymin + frac * (ymax - ymin)], # lower left
        [xmin + (1 - frac) * (xmax - xmin), ymin + frac * (ymax - ymin)], # lower right 
        [xmin + frac * (xmax - xmin), ymin + (1 - frac) * (ymax - ymin)], # upper left
        [xmin + (1 - frac) * (xmax - xmin), ymin + (1 - frac) * (ymax - ymin)], # upper right
    ])

    rep_indices = []

    for target in targets:
        d = np.linalg.norm(cont_atom_pos - target[None, :], axis=1)
        nearest_local = np.argmin(d)
        rep_indices.append(continuum_indices[nearest_local])

    return np.array(rep_indices, dtype=int), targets


# 4 evenly spaced out lower-left, lower-right, upper-left, upper-right
def pick_manual_rep_atoms_piecewise(atom_pos, continuum_indices,frac=0.25):
    cont_atom_pos = atom_pos[continuum_indices]

    xmin, ymin = cont_atom_pos.min(axis=0)
    xmax, ymax = cont_atom_pos.max(axis=0)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    # One center + four evenly distributed interior targets
    targets = np.array([
        [xmin + frac * (xmax - xmin), ymin + frac * (ymax - ymin)], # lower left
        [xmin + (1 - frac) * (xmax - xmin), ymin + frac * (ymax - ymin)], # lower right 
        [xmin + frac * (xmax - xmin), ymin + (1 - frac) * (ymax - ymin)], # upper left
        [xmin + (1 - frac) * (xmax - xmin), ymin + (1 - frac) * (ymax - ymin)], # upper right
    ])

    rep_indices = []

    for target in targets:
        d = np.linalg.norm(cont_atom_pos - target[None, :], axis=1)
        nearest_local = np.argmin(d)
        rep_indices.append(continuum_indices[nearest_local])

    return np.array(rep_indices, dtype=int), targets




################################################################
############## CODE FOR 2D LATTICE WITH DEFECT #################
################################################################

""" 
    shared code for 2D problems WITH DEFECTS. 
    Only for problems WIT DEFECTS!!
    
    split_atomistic_continuum_with_defect 
        outermost 3 layers of atoms are atomistic. 
        3 layers around the defect are atomistic
        rest are all continuum 
    
    pick_manual_rep_atoms_with_defect
        8 rep-atoms are distributed uniformly N,S,E,W,NW,NE,SW,SE
    
    THIS IS ONLY FOR PROBLEMS WITH DEFECTS
        problems without defects will have different split_atomistic_continuum
        problems without defects iwll have different pick_manual_rep_atoms
     
"""


def remove_central_atom(atom_pos):
    """
    creates a defect. remove the atom closest to the geometric center.

    Returns:
        atom_pos_defect: positions after removing one atom
        defect_pos: original position of removed atom
        defect_old_idx: index of removed atom in original full lattice
    """
    center = atom_pos.mean(axis=0)

    dist = np.linalg.norm(atom_pos - center[None, :], axis=1)
    defect_old_idx = np.argmin(dist)

    defect_pos = atom_pos[defect_old_idx].copy()

    keep_mask = np.ones(len(atom_pos), dtype=bool)
    keep_mask[defect_old_idx] = False

    atom_pos_defect = atom_pos[keep_mask]

    return atom_pos_defect, defect_pos, defect_old_idx



def split_atomistic_continuum_defect(atom_pos_defect, defect_pos, nx=49, ny=50, spacing=2**(1/6),border_layers=3, defect_layers=3):
    """
        find the indices of atoms that are in the vicinity of defect or on the border
        the defcet_region is 3 atoms deep
        the border layer is 3 atoms deep. 
        usng flattened index:index = j * nx + i
        split atoms into:
            fully atomistic atoms - first/last border_layers rows/columns
            fully atomistic atoms - first defect_layers rows/columns around defect_idx
            continuum atoms       - remaining interior atoms    
    """
    atomistic_indices = []
    continuum_indices = []

    # Triangular lattice row spacing. this is the "thickness" of each row so it's used for y-values (not x-values)
    row_spacing = (np.sqrt(3) / 2) * spacing

    for idx in range(len(atom_pos_defect)):
        pos_x,pos_y = atom_pos_defect[idx]
        is_border_atom = (
            pos_x < spacing*border_layers or 
            pos_x >= spacing*(nx - border_layers) or 
            pos_y < row_spacing*border_layers or 
            pos_y >= row_spacing*(ny - border_layers) 
        )

        dist_from_defect = np.linalg.norm(defect_pos - atom_pos_defect[idx])
        is_defect_region_atom = dist_from_defect <= spacing*defect_layers + 1e-10

        if is_border_atom or is_defect_region_atom:
            atomistic_indices.append(idx)
        else:
            continuum_indices.append(idx)

    return (
        np.array(atomistic_indices, dtype=int),
        np.array(continuum_indices, dtype=int),
    )



def pick_manual_rep_atoms_defect(atom_pos_defect, continuum_indices,frac=0.25):
    """
        place 8 rep-atoms on the square 
        evenly spaced out lower-left, lower-right, upper-left, upper-right, centre-left, center-right, center-top, center-bottom
    """
    continuum_atom_pos_defect = atom_pos_defect[continuum_indices]

    xmin, ymin = continuum_atom_pos_defect.min(axis=0)
    xmax, ymax = continuum_atom_pos_defect.max(axis=0)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    x_width = xmax - xmin
    y_width = ymax - ymin

    # One center + four evenly distributed interior targets
    targets = np.array([
        [xmin + frac * x_width,       ymin + frac * y_width], # lower left
        [xmin + (1 - frac) * x_width, ymin + frac * y_width], # lower right 
        [xmin + frac * x_width,       ymin + (1 - frac) * y_width], # upper left
        [xmin + (1 - frac) * x_width, ymin + (1 - frac) * y_width], # upper right
        
        [cx, ymin + frac * y_width], # mid-left
        [cx, ymin + (1 - frac) * y_width], # mid-right
        [xmin + frac * x_width,       cy], # mid-bottom left
        [xmin + (1-frac) * x_width,       cy], # mid-top
        
    ])

    rep_indices = []

    for target in targets:
        d = np.linalg.norm(continuum_atom_pos_defect - target[None, :], axis=1)
        nearest_local = np.argmin(d)
        rep_indices.append(continuum_indices[nearest_local])

    return np.array(rep_indices, dtype=int), targets



"""
############################################################
############# COMPUTE THE DEFORMATION GRADIENT #############
############################################################
"""

def local_deformation_gradient(
    reference_pos,
    current_pos,
    neighs,
    min_neighbors=2,
    eps=1e-12,
):
    """
    Compute local best-fit deformation gradient F_i for every atom.

    Works for defective lattices as long as each atom has enough surviving
    non-collinear neighbors.
    """
    dim = reference_pos.shape[1]
    I = np.eye(dim)

    F_all = np.zeros((len(reference_pos), dim, dim), dtype=np.float64)
    q_full = np.zeros(len(reference_pos), dtype=np.float64)
    valid = np.zeros(len(reference_pos), dtype=bool)

    for i in range(len(reference_pos)):
        nbrs = neighs[i]

        if len(nbrs) < min_neighbors:
            F_all[i] = I
            q_full[i] = eps
            valid[i] = False
            continue

        D = (reference_pos[nbrs] - reference_pos[i]).T
        d = (current_pos[nbrs] - current_pos[i]).T

        A = D @ D.T

        # Check rank. In 2D, need rank 2.
        if np.linalg.matrix_rank(A) < dim:
            F_all[i] = I
            q_full[i] = eps
            valid[i] = False
            continue

        F = d @ D.T @ np.linalg.pinv(A)

        F_all[i] = F
        q_full[i] = np.linalg.norm(F - I, ord="fro") + eps
        valid[i] = True

    return F_all, q_full, valid


"""
############################################################
################### RECLUSTERING HELPERS ###################
############################################################
"""

def scalar_field_quadrature_error(
    values_full,
    continuum_indices,
    rep_indices,
    cac_weights,
    eps=1e-12,
):
    """
    Compute the error in some scalar-field over the continuum atoms
    error = difference between true-continuum-field and repatom-weighted-field

    values_full: the scalar-field over the continuum atoms
        NumPy array, shape (N_atoms,)

    continuum_indices: global atom indices of continuum atoms

    rep_indices: global atom indices of representative atoms

    weights: CAC weights, number of atoms in each cluster

    Returns
    -------
    rel_error, full_value, reduced_value
    """
    values_full = np.asarray(values_full, dtype=np.float64)

    full_value = np.sum(values_full[continuum_indices])
    reduced_value = np.sum(cac_weights * values_full[rep_indices])

    rel_error = np.abs(full_value - reduced_value) / (np.abs(full_value) + eps)

    return rel_error, full_value, reduced_value



def split_clusters_at_defgrad_discontinuities(
    atom_pos,
    continuum_indices,
    cluster_idx,
    F_full,
    valid_full,
    neighs,
    importance_weights=None,
    jump_tol=5.0e-2,
    mad_factor=6.0,
    min_cluster_size=10,
    min_cut_edges=3,
):
    """
    Split clusters along deformation-gradient discontinuities.

    Neighbor edges satisfying

        ||F_i - F_j||_F > threshold

    are removed. Connected components of the remaining graph become
    separate clusters.

    Parameters
    ----------
    atom_pos
        Current or reference positions of all atoms.

    continuum_indices
        Global atom indices corresponding to cluster_idx.

    cluster_idx
        Cluster labels for continuum atoms.

    F_full
        Local deformation-gradient tensor for every atom.

    valid_full
        Boolean indicating whether each deformation gradient is valid.

    neighs
        Reference first-neighbor list using global atom indices.

    importance_weights
        Optional weights used to calculate cluster centers. CAC weights
        remain the number of atoms in each final cluster.

    Returns
    -------
    cluster_idx_new, centers_new, rep_indices_new, cac_weights_new,
    split_info
    """
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    continuum_indices = np.asarray(continuum_indices, dtype=int)
    cluster_idx = np.asarray(cluster_idx, dtype=int)

    n_continuum = len(continuum_indices)

    if len(cluster_idx) != n_continuum:
        raise ValueError(
            "cluster_idx must have length len(continuum_indices)"
        )

    if importance_weights is None:
        importance_weights = np.ones(n_continuum, dtype=np.float64)
    else:
        importance_weights = np.asarray(
            importance_weights,
            dtype=np.float64,
        )

    global_to_local = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(continuum_indices)
    }

    final_components = []
    split_records = []

    for old_cluster in np.unique(cluster_idx):
        members = np.where(cluster_idx == old_cluster)[0]

        if len(members) < 2 * min_cluster_size:
            final_components.append(members)
            continue

        member_set = set(members.tolist())
        cluster_edges = []
        jump_values = []

        for local_i in members:
            global_i = continuum_indices[local_i]

            for global_j in neighs[global_i]:
                local_j = global_to_local.get(int(global_j))

                if local_j is None or local_j not in member_set:
                    continue

                # Count each undirected edge only once.
                if local_j <= local_i:
                    continue

                if not (valid_full[global_i] and valid_full[global_j]):
                    continue

                jump = np.linalg.norm(
                    F_full[global_i] - F_full[global_j],
                    ord="fro",
                )

                cluster_edges.append((local_i, local_j, jump))
                jump_values.append(jump)

        if not cluster_edges:
            final_components.append(members)
            continue

        jump_values = np.asarray(jump_values)

        jump_median = np.median(jump_values)
        jump_mad = np.median(np.abs(jump_values - jump_median))

        robust_jump_tol = (
            jump_median
            + mad_factor * 1.4826 * jump_mad
        )

        effective_jump_tol = max(jump_tol, robust_jump_tol)

        smooth_adjacency = {
            local_idx: []
            for local_idx in members
        }

        cut_edges = []

        for local_i, local_j, jump in cluster_edges:
            if jump > effective_jump_tol:
                cut_edges.append((local_i, local_j, jump))
            else:
                smooth_adjacency[local_i].append(local_j)
                smooth_adjacency[local_j].append(local_i)

        if len(cut_edges) < min_cut_edges:
            final_components.append(members)
            continue

        # Find connected components after removing discontinuity edges.
        unvisited = set(members.tolist())
        components = []

        while unvisited:
            start = unvisited.pop()
            stack = [start]
            component = [start]

            while stack:
                local_i = stack.pop()

                for local_j in smooth_adjacency[local_i]:
                    if local_j in unvisited:
                        unvisited.remove(local_j)
                        stack.append(local_j)
                        component.append(local_j)

            components.append(np.asarray(component, dtype=int))

        large_components = [
            component
            for component in components
            if len(component) >= min_cluster_size
        ]

        # A meaningful split needs at least two sufficiently large sides.
        if len(large_components) < 2:
            final_components.append(members)
            continue

        small_components = [
            component
            for component in components
            if len(component) < min_cluster_size
        ]

        # Merge tiny isolated fragments into the nearest large component.
        large_components = [
            component.copy()
            for component in large_components
        ]

        large_centers = np.asarray([
            atom_pos[continuum_indices[component]].mean(axis=0)
            for component in large_components
        ])

        for small_component in small_components:
            small_center = atom_pos[
                continuum_indices[small_component]
            ].mean(axis=0)

            nearest = np.argmin(
                np.linalg.norm(
                    large_centers - small_center[None, :],
                    axis=1,
                )
            )

            large_components[nearest] = np.concatenate(
                [large_components[nearest], small_component]
            )

            large_centers[nearest] = atom_pos[
                continuum_indices[large_components[nearest]]
            ].mean(axis=0)

        final_components.extend(large_components)

        split_records.append(
            {
                "old_cluster": int(old_cluster),
                "old_size": int(len(members)),
                "new_sizes": [
                    int(len(component))
                    for component in large_components
                ],
                "jump_threshold": float(effective_jump_tol),
                "max_jump": float(jump_values.max()),
                "num_cut_edges": int(len(cut_edges)),
            }
        )

    n_new_clusters = len(final_components)

    cluster_idx_new = np.empty(n_continuum, dtype=int)
    centers_new = np.zeros(
        (n_new_clusters, atom_pos.shape[1]),
        dtype=np.float64,
    )
    rep_indices_new = np.zeros(n_new_clusters, dtype=int)
    cac_weights_new = np.zeros(n_new_clusters, dtype=np.float64)

    for new_cluster, members in enumerate(final_components):
        cluster_idx_new[members] = new_cluster

        global_members = continuum_indices[members]
        positions = atom_pos[global_members]

        q = np.maximum(importance_weights[members], 1e-12)

        center = np.sum(q[:, None] * positions, axis=0) / np.sum(q)

        centers_new[new_cluster] = center
        cac_weights_new[new_cluster] = len(members)

        nearest_member = np.argmin(
            np.linalg.norm(positions - center[None, :], axis=1)
        )

        rep_indices_new[new_cluster] = global_members[nearest_member]

    split_info = {
        "did_split": bool(split_records),
        "num_clusters_before": int(len(np.unique(cluster_idx))),
        "num_clusters_after": int(n_new_clusters),
        "splits": split_records,
    }

    return (
        cluster_idx_new,
        centers_new,
        rep_indices_new,
        cac_weights_new,
        split_info,
    )

def recluster_deformation_gradient_weighted(
    atom_pos_ref,
    current_pos_np,
    continuum_indices,
    cluster_idx,
    rep_indices,
    cac_weights,
    neighs,
    trigger_tol=1.0e-2,
    jump_tol=1.0e-4,
    jump_mad_factor=6.0,
    min_split_cluster_size=10.0,
    min_cut_edges=3.0,
    max_iter=100,
    tol=1e-10,
):
    F_full, q_defgrad_full, valid_full = local_deformation_gradient(
        reference_pos=atom_pos_ref,
        current_pos=current_pos_np,
        neighs=neighs,
    )

    rel_error, full_value, reduced_value = (
        scalar_field_quadrature_error(
            values_full=q_defgrad_full,
            continuum_indices=continuum_indices,
            rep_indices=rep_indices,
            cac_weights=cac_weights,
        )
    )

    q_continuum = q_defgrad_full[continuum_indices]

    # Check whether any existing cluster crosses a discontinuity.
    (
        split_cluster_idx,
        split_centers,
        split_rep_indices,
        split_cac_weights,
        initial_split_info,
    ) = split_clusters_at_defgrad_discontinuities(
        atom_pos=current_pos_np,
        continuum_indices=continuum_indices,
        cluster_idx=cluster_idx,
        F_full=F_full,
        valid_full=valid_full,
        neighs=neighs,
        importance_weights=q_continuum,
        jump_tol=jump_tol,
        mad_factor=jump_mad_factor,
        min_cluster_size=min_split_cluster_size,
        min_cut_edges=min_cut_edges,
    )

    quadrature_triggered = rel_error > trigger_tol
    discontinuity_triggered = initial_split_info["did_split"]

    result = {
        "F_full": F_full,
        "q_defgrad_full": q_defgrad_full,
        "valid_defgrad_full": valid_full,
        "rel_defgrad_quad_error": rel_error,
        "full_defgrad_value": full_value,
        "reduced_defgrad_value": reduced_value,
        "quadrature_triggered": quadrature_triggered,
        "discontinuity_triggered": discontinuity_triggered,
    }

    if not quadrature_triggered and not discontinuity_triggered:
        return False, result

    if quadrature_triggered:
        (
            base_cluster_idx,
            base_centers,
            base_rep_indices,
            base_cac_weights,
            history,
        ) = kmeans_weighted(
            atom_pos=current_pos_np,
            continuum_indices=continuum_indices,
            initial_rep_indices=rep_indices,
            importance_weights=q_continuum,
            max_iter=max_iter,
            tol=tol,
        )

        # Weighted K-means may still place a cluster across the jump.
        (
            cluster_idx_new,
            centers_new,
            rep_indices_new,
            cac_weights_new,
            split_info,
        ) = split_clusters_at_defgrad_discontinuities(
            atom_pos=current_pos_np,
            continuum_indices=continuum_indices,
            cluster_idx=base_cluster_idx,
            F_full=F_full,
            valid_full=valid_full,
            neighs=neighs,
            importance_weights=q_continuum,
            jump_tol=jump_tol,
            mad_factor=jump_mad_factor,
            min_cluster_size=min_split_cluster_size,
            min_cut_edges=min_cut_edges,
        )
    else:
        cluster_idx_new = split_cluster_idx
        centers_new = split_centers
        rep_indices_new = split_rep_indices
        cac_weights_new = split_cac_weights
        split_info = initial_split_info
        history = []

    result.update(
        {
            "cluster_idx": cluster_idx_new,
            "centers": centers_new,
            "rep_indices": rep_indices_new,
            "cac_weights": cac_weights_new,
            "history": history,
            "split_info": split_info,
        }
    )

    return True, result


