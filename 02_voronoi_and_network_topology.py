#!/usr/bin/env python3
"""
Step 2: Voronoi Tessellation & Network Topology Analysis
========================================================
Part of the Hands-on Hyperuniformity Laboratory (Mint Leaf Analysis).

This script performs:
  1. Loads areole centroids and geometric properties from 'areoles_and_centroids.npz'.
  2. Nearest-neighbor distance statistics via KD-Tree.
  3. Delaunay triangulation and dual Voronoi cellular foam graph.
  4. Coordination number distribution P(z) for interior nodes (evaluating the
     Euler trivalent foam law <z> ~ 6 and 5-7 disclination defect pairs).
  5. Lognormal fitting of the interior areole area distribution.
  6. Exports:
       - 'figures/fig3_tessellation_coordination.png' & '.pdf'
       - 'topology_summary.json'
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Delaunay, Voronoi, voronoi_plot_2d
from scipy.stats import lognorm
import json
import os

# Publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

os.makedirs("figures", exist_ok=True)

print("=" * 70)
print("STEP 2: Voronoi Tessellation & Network Topology Analysis")
print("=" * 70)

# 1. Load intermediate data from Step 1
archive_path = "areoles_and_centroids.npz"
if not os.path.exists(archive_path):
    raise FileNotFoundError(
        f"'{archive_path}' not found! Please run '01_skeletonize_and_extract_centroids.py' first."
    )

print(f"\n[1/4] Loading centroids from '{archive_path}'...")
data = np.load(archive_path)
centroids = data["centroids"]
areas = data["areas"]
is_interior = data["is_interior"]
N = len(centroids)
N_interior = int(np.sum(is_interior))
print(f"      Loaded {N} total centroids ({N_interior} interior, {N - N_interior} boundary).")

# 2. Nearest-Neighbor Spacing via KD-Tree
print("\n[2/4] Computing nearest-neighbor distances via spatial KD-Tree...")
tree = KDTree(centroids)
nn_dists, _ = tree.query(centroids, k=2)
nn_dists = nn_dists[:, 1]
mean_nn = float(np.mean(nn_dists))
std_nn = float(np.std(nn_dists))
min_nn = float(np.min(nn_dists))
max_nn = float(np.max(nn_dists))
print(f"      Mean NN distance: {mean_nn:.2f} ± {std_nn:.2f} px")
print(f"      Min / Max NN distance: {min_nn:.2f} / {max_nn:.2f} px")

# 3. Delaunay Triangulation & Coordination Numbers
print("\n[3/4] Constructing Delaunay triangulation & dual Voronoi graph...")
tri = Delaunay(centroids)
vor = Voronoi(centroids)

neighbors = [set() for _ in range(N)]
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            neighbors[simplex[i]].add(simplex[j])
            neighbors[simplex[j]].add(simplex[i])

coordination_numbers = np.array([len(nbrs) for nbrs in neighbors])

# Critical pedagogical step: filter exclusively interior nodes to eliminate
# spurious boundary degree truncation!
coord_interior = coordination_numbers[is_interior]
unique_coords, coord_counts = np.unique(coord_interior, return_counts=True)
coord_probs = coord_counts / len(coord_interior)

mean_coord = float(np.mean(coord_interior))
hex_fraction = float(np.sum(coord_interior == 6) / len(coord_interior))
p5_fraction = float(np.sum(coord_interior == 5) / len(coord_interior))
p7_fraction = float(np.sum(coord_interior == 7) / len(coord_interior))

print("\n" + "-" * 55)
print("  TOPOLOGICAL FOAM SUMMARY (Euler Trivalent Constraint):")
print("-" * 55)
print(f"  Mean coordination <z>  : {mean_coord:.3f}  (Theory: <z> = 6.000)")
print(f"  Hexagonal fraction P(6): {hex_fraction * 100:.1f}%")
print(f"  Pentagonal fraction P(5): {p5_fraction * 100:.1f}%  (negative disclinations)")
print(f"  Heptagonal fraction P(7): {p7_fraction * 100:.1f}%  (positive disclinations)")
print("-" * 55)
print("  Note: Balanced 5-7 disclination dipoles preserve space-filling")
print("  compliance while keeping the ensemble average <z> ~ 6.")

# 4. Generate Figure 3 & Export Results
print("\n[4/4] Plotting Figure 3 (Delaunay graph, P(z), and lognormal area fit)...")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.4))

# Panel a: Delaunay Triangulation
axes3[0].triplot(centroids[:, 0], centroids[:, 1], tri.simplices, color="#2b5c8f", lw=0.7, alpha=0.7)
axes3[0].plot(centroids[is_interior, 0], centroids[is_interior, 1], 'o', color="#d95f02", markersize=3, label="Interior")
axes3[0].plot(centroids[~is_interior, 0], centroids[~is_interior, 1], 'x', color="#7570b3", markersize=3, label="Boundary")
axes3[0].set_title("(a) Delaunay Triangulation", fontweight="bold")
axes3[0].set_aspect("equal")
axes3[0].axis("off")
axes3[0].legend(loc="lower right", framealpha=0.8)

# Panel b: Coordination distribution
bars = axes3[1].bar(unique_coords, coord_probs, color="#1b9e77", edgecolor="black", width=0.6, alpha=0.85)
axes3[1].set_xlabel("Coordination Number $z$")
axes3[1].set_ylabel("Probability $P(z)$")
axes3[1].set_title(f"(b) Coordination Distribution ($z=6$: {hex_fraction*100:.1f}%)", fontweight="bold")
axes3[1].set_xticks(unique_coords)
axes3[1].grid(True, linestyle="--", alpha=0.4)
for bar in bars:
    yval = bar.get_height()
    axes3[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
axes3[1].set_ylim(0, max(coord_probs) + 0.08)

# Panel c: Lognormal area fit
areas_interior = areas[is_interior]
axes3[2].hist(areas_interior, bins=20, density=True, color="#386cb0", edgecolor="black", alpha=0.75, label="Data")
shape, loc, scale = lognorm.fit(areas_interior, floc=0)
x_fit = np.linspace(min(areas_interior), max(areas_interior), 200)
axes3[2].plot(x_fit, lognorm.pdf(x_fit, shape, loc, scale), 'r-', lw=2, label="Lognormal Fit")
axes3[2].set_xlabel(r"Areole Area $A$ [$\mathrm{px}^2$]")
axes3[2].set_ylabel("Probability Density")
axes3[2].set_title("(c) Areole Area Distribution", fontweight="bold")
axes3[2].grid(True, linestyle="--", alpha=0.4)
axes3[2].legend()

plt.tight_layout()
fig3.savefig("figures/fig3_tessellation_coordination.png", dpi=300)
fig3.savefig("figures/fig3_tessellation_coordination.pdf")
plt.close(fig3)
print("      Saved 'figures/fig3_tessellation_coordination.png' and '.pdf'")

topology_summary = {
    "total_centroids": N,
    "interior_centroids": N_interior,
    "mean_nn_dist": mean_nn,
    "std_nn_dist": std_nn,
    "min_nn_dist": min_nn,
    "max_nn_dist": max_nn,
    "mean_coordination": mean_coord,
    "hexagonal_fraction": hex_fraction,
    "pentagonal_fraction": p5_fraction,
    "heptagonal_fraction": p7_fraction,
    "coordination_distribution": {int(k): float(v) for k, v in zip(unique_coords, coord_probs)},
    "lognormal_fit": {
        "shape_sigma": float(shape),
        "scale_exp_mu": float(scale),
        "mu_log": float(np.log(scale))
    }
}

with open("topology_summary.json", "w") as f:
    json.dump(topology_summary, f, indent=4)
print("      Saved 'topology_summary.json'")

print("\n" + "=" * 70)
print("Step 2 complete! You can now run '03_hyperuniformity_fluctuations.py'.")
print("=" * 70)
