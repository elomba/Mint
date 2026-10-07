#!/usr/bin/env python3
"""
Step 1: Leaf Segmentation, Skeletonization, and Centroid/Area Extraction
=======================================================================
Part of the Hands-on Hyperuniformity Laboratory (Mint Leaf Analysis).

This script performs:
  1. Image loading and CIELAB color-space leaf blade segmentation.
  2. Multi-scale Black Top-Hat morphological filtering + Sobel gradient to
     enhance dark groove vein networks into a topographic elevation surface.
  3. Marker-controlled watershed segmentation to partition the lamina into
     watertight photosynthetic areole polygons.
  4. 1-pixel medial axis skeletonization of the vein network.
  5. Geometric centroid localization, area, perimeter, and boundary classification.
  6. Exports:
       - 'figures/fig1_vein_segmentation.png' & '.pdf'
       - 'figures/fig2_areole_polygons.png' & '.pdf'
       - 'areoles_and_centroids.npz' (NumPy archive for subsequent steps)
       - 'areoles_data.csv' (Tabular format for student inspection)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
print("STEP 1: Leaf Segmentation, Skeletonization & Centroid Extraction")
print("=" * 70)

# 1. Load image and compute clean leaf mask
print("\n[1/5] Loading 'Mint.jpg' and computing binary leaf mask...")
if not os.path.exists("Mint.jpg"):
    raise FileNotFoundError("Could not find 'Mint.jpg' in current directory.")

img = cv2.imread('Mint.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L_channel = lab[:, :, 0].astype(np.float32)
a_channel = lab[:, :, 1]

# Binary mask: a* < 120 separates green leaf from ambient wood background
raw_mask = (a_channel < 120).astype(np.uint8) * 255
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
leaf_clean = (labels == largest_label).astype(np.uint8) * 255

# Fill interior holes
contours, _ = cv2.findContours(leaf_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_mask = np.zeros_like(leaf_clean)
cv2.drawContours(leaf_mask, contours, -1, 255, -1)
leaf_bool = leaf_mask > 0
leaf_area_px = int(np.sum(leaf_bool))
dist_map = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)

print(f"      Frame size: {w} x {h} px")
print(f"      Segmented leaf area: {leaf_area_px:,} px ({leaf_area_px / (h * w) * 100:.2f}% of total frame)")

# 2. Vein contrast enhancement and elevation surface
print("\n[2/5] Enhancing veins via multi-scale Black Top-Hat & Sobel gradient...")
L_smooth = cv2.GaussianBlur(L_channel, (0, 0), 2.2)

# Multi-scale black top hat for dark grooves
bth = np.zeros_like(L_channel)
for k in [7, 11, 15, 21]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bth = np.maximum(bth, cv2.morphologyEx(L_smooth.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel))

grad_x = cv2.Sobel(L_smooth, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(L_smooth, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)

elevation = bth.astype(np.float32) + 0.5 * grad_mag
elevation[~leaf_bool] = elevation.max()

areole_score = L_smooth - 0.8 * bth.astype(np.float32)
valid_region = leaf_bool & (dist_map > 8)

peaks = peak_local_max(areole_score, min_distance=15, labels=valid_region)
num_peaks = len(peaks)
print(f"      Detected {num_peaks} initial regional seeds for watershed.")

# 3. Marker-Controlled Watershed Segmentation
print("\n[3/5] Flooding topographic elevation surface (Watershed)...")
markers = np.zeros(L_channel.shape, dtype=np.int32)
for idx, (r, c) in enumerate(peaks, 1):
    markers[r, c] = idx

ws = watershed(elevation, markers, mask=leaf_bool)

# 4. Extract Vein Skeleton & Polygon Centroids
print("\n[4/5] Computing 1-pixel medial axis skeleton and extracting areole polygons...")
edges = np.zeros_like(leaf_bool, dtype=bool)
for sr, sc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    rolled = np.roll(ws, (sr, sc), axis=(0, 1))
    edges |= (ws != rolled) & leaf_bool & np.roll(leaf_bool, (sr, sc), axis=(0, 1))

skel = skeletonize(edges)
total_skel_length = int(np.sum(skel))

polygon_centroids = []
polygon_areas = []
polygon_perimeters = []
polygon_circularities = []
is_interior = []

leaf_boundary = cv2.Canny(leaf_mask, 100, 200) > 0

for idx in range(1, num_peaks + 1):
    pm = (ws == idx)
    area = np.sum(pm)
    if area < 10:
        continue
    ys, xs = np.where(pm)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    
    # Boundary touch check
    touches = np.any(pm & leaf_boundary)
    
    # Contour for perimeter
    cnts, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(cnts[0], True) if len(cnts) > 0 else 0.0
    circ = (4 * np.pi * area) / (peri**2 + 1e-6) if peri > 0 else 0.0

    polygon_centroids.append([cx, cy])
    polygon_areas.append(int(area))
    polygon_perimeters.append(float(peri))
    polygon_circularities.append(float(min(1.0, circ)))
    is_interior.append(not bool(touches))

polygon_centroids = np.array(polygon_centroids)
polygon_areas = np.array(polygon_areas)
polygon_perimeters = np.array(polygon_perimeters)
polygon_circularities = np.array(polygon_circularities)
is_interior = np.array(is_interior)

N = len(polygon_centroids)
N_interior = int(np.sum(is_interior))
N_boundary = N - N_interior
eq_diameters = 2.0 * np.sqrt(polygon_areas / np.pi)

print(f"      Total areole polygons: {N}")
print(f"      Interior areoles (unclipped): {N_interior}")
print(f"      Boundary areoles (edge-clipped): {N_boundary}")
print(f"      Vein skeleton cumulative length: {total_skel_length:,} px")
print(f"      Mean areole area: {np.mean(polygon_areas):.1f} ± {np.std(polygon_areas):.1f} px² (median: {np.median(polygon_areas):.1f} px²)")

# 5. Save Visualizations & Data Archives
print("\n[5/5] Generating figures and exporting intermediate data archives...")

# --- FIGURE 1: Vein Segmentation ---
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))
axes1[0, 0].imshow(img_rgb)
axes1[0, 0].set_title("(a) Original Mint Leaf Photograph", fontweight="bold")
axes1[0, 0].axis("off")

axes1[0, 1].imshow(leaf_mask, cmap="gray")
axes1[0, 1].set_title(f"(b) Segmented Leaf Blade Mask (Area: {leaf_area_px:,} px)", fontweight="bold")
axes1[0, 1].axis("off")

elev_disp = elevation.copy()
elev_disp[~leaf_bool] = 0
axes1[1, 0].imshow(elev_disp, cmap="inferno")
axes1[1, 0].set_title("(c) Multi-Scale Groove & Gradient Elevation Surface", fontweight="bold")
axes1[1, 0].axis("off")

vis_skel = img_rgb.copy()
skel_dilated = cv2.dilate(skel.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
vis_skel[skel_dilated] = [255, 30, 30]
axes1[1, 1].imshow(vis_skel)
axes1[1, 1].set_title(f"(d) Skeletonized Vein Network (Length: {total_skel_length:,} px)", fontweight="bold")
axes1[1, 1].axis("off")

plt.tight_layout()
fig1.savefig("figures/fig1_vein_segmentation.png", dpi=300)
fig1.savefig("figures/fig1_vein_segmentation.pdf")
plt.close(fig1)
print("      Saved 'figures/fig1_vein_segmentation.png' and '.pdf'")

# --- FIGURE 2: Areole Polygons & Centroids ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.8))
rng_colors = np.random.default_rng(101)
poly_colors = rng_colors.uniform(0.3, 0.95, size=(num_peaks + 1, 3))
colored_polys = np.ones((h, w, 3), dtype=np.float32)
for idx in range(1, num_peaks + 1):
    colored_polys[ws == idx] = poly_colors[idx]
colored_polys[~leaf_bool] = [0.12, 0.12, 0.12]
colored_polys[skel] = [0.0, 0.0, 0.0]

axes2[0].imshow(colored_polys)
axes2[0].plot(polygon_centroids[:, 0], polygon_centroids[:, 1], 'k.', markersize=2.5, alpha=0.8)
axes2[0].set_title(f"(a) Polygonal Areole Partition ({N} Polygons)", fontweight="bold")
axes2[0].axis("off")

vis_cents = img_rgb.copy()
vis_cents[skel_dilated] = [255, 230, 0] # Bright yellow veins
for cx_pt, cy_pt in polygon_centroids:
    cv2.circle(vis_cents, (int(round(cx_pt)), int(round(cy_pt))), 4, (0, 220, 255), -1)
    cv2.circle(vis_cents, (int(round(cx_pt)), int(round(cy_pt))), 1, (0, 0, 0), -1)

rx1, rx2, ry1, ry2 = 520, 760, 260, 460
cv2.rectangle(vis_cents, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

axes2[1].imshow(vis_cents)
axes2[1].set_title(f"(b) Located Centroids on Mint Leaf (N={N})", fontweight="bold")
axes2[1].axis("off")

ax_ins = inset_axes(axes2[1], width="32%", height="32%", loc="lower right", borderpad=1.2)
ax_ins.imshow(vis_cents[ry1:ry2, rx1:rx2])
ax_ins.set_xticks([])
ax_ins.set_yticks([])
for spine in ax_ins.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)
ax_ins.set_title("Close-up Detail", color="white", fontsize=9, pad=3)

fig2.savefig("figures/fig2_areole_polygons.png", dpi=300, bbox_inches="tight")
fig2.savefig("figures/fig2_areole_polygons.pdf", bbox_inches="tight")
plt.close(fig2)
print("      Saved 'figures/fig2_areole_polygons.png' and '.pdf'")

# Save NumPy archive
np.savez_compressed(
    "areoles_and_centroids.npz",
    centroids=polygon_centroids,
    areas=polygon_areas,
    perimeters=polygon_perimeters,
    circularities=polygon_circularities,
    eq_diameters=eq_diameters,
    is_interior=is_interior,
    leaf_mask=leaf_mask,
    dist_map=dist_map,
    skel=skel,
    leaf_area_px=leaf_area_px,
    h=h,
    w=w
)
print("      Exported 'areoles_and_centroids.npz' (NumPy archive for Steps 2 and 3)")

# Save tabular CSV for student exploration
with open("areoles_data.csv", "w") as f:
    f.write("id,centroid_x,centroid_y,area_px2,perimeter_px,circularity,eq_diameter_px,is_interior\n")
    for i in range(N):
        f.write(f"{i+1},{polygon_centroids[i, 0]:.2f},{polygon_centroids[i, 1]:.2f},"
                f"{polygon_areas[i]},{polygon_perimeters[i]:.2f},{polygon_circularities[i]:.4f},"
                f"{eq_diameters[i]:.2f},{int(is_interior[i])}\n")
print("      Exported 'areoles_data.csv' (Tabular CSV)")

print("\n" + "=" * 70)
print("Step 1 complete! You can now run '02_voronoi_and_network_topology.py'.")
print("=" * 70)
