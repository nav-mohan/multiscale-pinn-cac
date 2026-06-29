# Simulation Error Analysis & Performance Metrics

> **Simulation Configuration:**
> * `relative-error-tolerance` = `1e-3`
> * `absolute-error-tolerance` = `1e-6`

---

## 1. Structural Metric Errors (REDUCED vs. LAMMPS)

### Bond-Length Error
![Bond-Length Error Comparison](bond-length-1e3-1e6.png)

### Displacement-Field Error
![Displacement-Field Error Comparison](displacement-1e3-1e6.png)

---

## 2. Neural Network Accuracy (REDUCED vs. LAMMPS)

### Per-Atom Energy Difference
Comparison of the per-atom energy predicted by the `REDUCED` model against the baseline prediction from `LAMMPS`.

![Per-Atom Energy Difference (REDUCED minus LAMMPS)](pinn_minus_lammps_atom_energy-REL1e3-ABS1e6-TRIAL3.gif)

---

## 3. Comparative Analysis (REDUCED vs. FULL Simulation)

### Per-Atom Energy
| REDUCED System | FULL Simulation |
| :---: | :---: |
| ![Reduced Atom Energy](output-IL1-BL3-REL1e3-ABS1e6-TRIAL2-atom_energy.gif) | ![Full Atom Energy](output-IL1-BL3-full-TRIAL3-atom_energy.gif) |

### Deformation Gradient Magnitude ($|F_i - I|$)
| REDUCED System | FULL Simulation |
| :---: | :---: |
| ![Reduced q_full](output-IL1-BL3-REL1e3-ABS1e6-TRIAL3-q_full.gif) | ![Full q_full](output-IL1-BL3-full-TRIAL3-q_full.gif) |

### Deformation Gradient Discontinuity ($|F_i - F_j|$)
| REDUCED System | FULL Simulation |
| :---: | :---: |
| ![Reduced detF](output-IL1-BL3-REL1e3-ABS1e6-TRIAL3-detF.gif) | ![Full detF](output-IL1-BL3-full-TRIAL3-detF.gif) |

---

## 4. Multi-Trial Aggregated Performance (Averaged Over 50 Trials)

### Energy Error vs. Epoch
Tracks both the `loss_energy - lammps_energy` and `true_energy - lammps_energy` profiles of the system.

![Energy Error vs Epoch](energy_error_vs_epoch/energy_error_vs_epoch_IL1-BL3-REL1e3-ABS1e6.png)

### Growth of Error & Reclustering Triggers
This accumulated metric triggers system **reclustering**. It is calculated as a weighted average of the **energy-variance** and the **discontinuity in the deformation-gradient**.

![Growth of Error](relative_quadrature_error/relative_quadrature_error_avg_IL1_BL3_REL1e3_ABS1e6.png)