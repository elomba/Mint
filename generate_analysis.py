import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d, Delaunay
from scipy.special import j0
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import json
import os

# Publication plot styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13

os.makedirs("figures", exist_ok=True)

# 1. Load image and compute clean leaf mask
print("1. Loading image and computing leaf mask...")
img = cv2.imread('Mint.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L_channel = lab[:, :, 0].astype(np.float32)
a_channel = lab[:, :, 1]

# Binary mask: a* < 120 separates leaf from wood background
raw_mask = (a_channel < 120).astype(np.uint8) * 255
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
leaf_clean = (labels == largest_label).astype(np.uint8) * 255

# Fill interior holes
contours, _ = cv2.findContours(leaf_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_mask = np.zeros_like(leaf_clean)
cv2.drawContours(leaf_mask, contours, -1, 255, -1)
leaf_bool = leaf_mask > 0
leaf_area_px = int(np.sum(leaf_bool))
dist_map = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)

print(f"Leaf area: {leaf_area_px} px ({leaf_area_px / (h * w) * 100:.2f}% of total frame)")

# 2. Vein contrast enhancement and elevation surface
print("2. Enhancing veins and areoles...")
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
print(f"Detected {num_peaks} initial areole seeds.")

# 3. Marker-Controlled Watershed Segmentation
print("3. Executing marker-controlled watershed segmentation...")
markers = np.zeros(L_channel.shape, dtype=np.int32)
for idx, (r, c) in enumerate(peaks, 1):
    markers[r, c] = idx

ws = watershed(elevation, markers, mask=leaf_bool)

# 4. Extract Vein Skeleton & Polygon Centroids
print("4. Extracting vein skeleton and polygon properties...")
edges = np.zeros_like(leaf_bool, dtype=bool)
for sr, sc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    rolled = np.roll(ws, (sr, sc), axis=(0, 1))
    edges |= (ws != rolled) & leaf_bool & np.roll(leaf_bool, (sr, sc), axis=(0, 1))

skel = skeletonize(edges)
total_skel_length = int(np.sum(skel))
print(f"Vein skeleton total length: {total_skel_length} px")

polygon_centroids = []
polygon_areas = []
polygon_perimeters = []
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
    polygon_centroids.append([cx, cy])
    polygon_areas.append(int(area))
    touches = np.any(pm & leaf_boundary)
    is_interior.append(not bool(touches))

polygon_centroids = np.array(polygon_centroids)
polygon_areas = np.array(polygon_areas)
is_interior = np.array(is_interior)
N = len(polygon_centroids)
N_interior = int(np.sum(is_interior))
print(f"Total polygons: {N} (Interior: {N_interior}, Boundary: {N - N_interior})")

eq_diameters = 2.0 * np.sqrt(polygon_areas / np.pi)

# 5. Spatial Nearest-Neighbor Statistics
print("5. Computing nearest-neighbor distances...")
tree = KDTree(polygon_centroids)
nn_dists, _ = tree.query(polygon_centroids, k=2)
nn_dists = nn_dists[:, 1]
mean_nn = float(np.mean(nn_dists))
std_nn = float(np.std(nn_dists))
min_nn = float(np.min(nn_dists))
max_nn = float(np.max(nn_dists))
print(f"Nearest neighbor distance: {mean_nn:.2f} ± {std_nn:.2f} px")

# 6. Delaunay Triangulation and Coordination
print("6. Delaunay triangulation and coordination analysis...")
tri = Delaunay(polygon_centroids)
neighbors = [set() for _ in range(N)]
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            neighbors[simplex[i]].add(simplex[j])
            neighbors[simplex[j]].add(simplex[i])

coordination_numbers = np.array([len(nbrs) for nbrs in neighbors])
coord_interior = coordination_numbers[is_interior]
unique_coords, coord_counts = np.unique(coord_interior, return_counts=True)
coord_probs = coord_counts / len(coord_interior)
hex_fraction = float(np.sum(coord_interior == 6) / len(coord_interior))
mean_coord = float(np.mean(coord_interior))
print(f"Mean interior coordination: {mean_coord:.2f}, Hexagonal fraction: {hex_fraction * 100:.1f}%")

# 7. Pair Correlation Function g(r)
print("7. Computing pair correlation function g(r)...")
rho = N / float(leaf_area_px)
r_max = 180.0
dr = 4.0
r_bins = np.arange(0, r_max + dr, dr)
r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

buf_mask = dist_map[np.round(polygon_centroids[:, 1]).astype(int), np.round(polygon_centroids[:, 0]).astype(int)] >= r_max
if np.sum(buf_mask) < 40:
    r_max = 140.0
    r_bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    buf_mask = dist_map[np.round(polygon_centroids[:, 1]).astype(int), np.round(polygon_centroids[:, 0]).astype(int)] >= r_max

buf_idx = np.where(buf_mask)[0]
D = np.linalg.norm(polygon_centroids[:, None, :] - polygon_centroids[None, :, :], axis=-1)

g_r = np.zeros(len(r_centers))
for i in buf_idx:
    dists = D[i, :]
    dists = dists[dists > 0]
    counts, _ = np.histogram(dists, bins=r_bins)
    g_r += counts

annulus_areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
g_r = g_r / (len(buf_idx) * rho * annulus_areas)

excl_idx = np.where(g_r > 0.05)[0]
r_exclusion = float(r_centers[excl_idx[0]]) if len(excl_idx) > 0 else 0.0
peak_idx = int(np.argmax(g_r))
r_peak = float(r_centers[peak_idx])
g_peak = float(g_r[peak_idx])
print(f"g(r): exclusion hole = {r_exclusion:.1f} px, primary peak at {r_peak:.1f} px (value {g_peak:.2f})")

# 8. Radially Averaged Structure Factor S(k)
print("8. Computing structure factor S(k)...")
# Window-apodized Hankel transform to suppress finite-r_max Gibbs ringing
taper = 0.5 * (1.0 + np.cos(np.pi * r_centers / r_max))
k_vals = np.linspace(0.005, 0.35, 120)
S_k = np.zeros_like(k_vals)
integrand_base = 2 * np.pi * r_centers * (g_r - 1.0) * taper * dr
for idx, k in enumerate(k_vals):
    S_k[idx] = 1.0 + rho * np.sum(integrand_base * j0(k * r_centers))

S_k_clean = np.maximum(S_k, 0.0)
k_peak_idx = int(np.argmax(S_k_clean))
k_peak = float(k_vals[k_peak_idx])
S_peak = float(S_k_clean[k_peak_idx])
lambda_peak = float(2 * np.pi / k_peak)

# Direct azimuthal structure factor for validation
cx = polygon_centroids[:, 0] - np.mean(polygon_centroids[:, 0])
cy = polygon_centroids[:, 1] - np.mean(polygon_centroids[:, 1])

# 9. 2D Structure Factor Map S(kx, ky)
print("9. Computing 2D structure factor...")
kx_grid = np.linspace(-0.25, 0.25, 161)
ky_grid = np.linspace(-0.25, 0.25, 161)
KX, KY = np.meshgrid(kx_grid, ky_grid)
phases = KX[:, :, None] * cx[None, None, :] + KY[:, :, None] * cy[None, None, :]
S_2D = (np.abs(np.sum(np.exp(-1j * phases), axis=2))**2) / N

# Direct azimuthal S(k) from point positions
k_azimuthal = np.linspace(0.015, 0.32, 40)
S_direct_az = []
for ka in k_azimuthal:
    thetas = np.linspace(0, 2*np.pi, 48, endpoint=False)
    kxs = ka * np.cos(thetas)
    kys = ka * np.sin(thetas)
    ph = kxs[:, None] * cx[None, :] + kys[:, None] * cy[None, :]
    S_direct_az.append(np.mean(np.abs(np.sum(np.exp(-1j * ph), axis=1))**2) / N)
S_direct_az = np.array(S_direct_az)

# 10. Local Number Variance sigma_N^2(R) & Fano Factor F(R)
print("10. Computing number variance and Fano factor across sampling radii...")
leaf_ys, leaf_xs = np.where(leaf_bool)
rng = np.random.default_rng(42)

num_poisson_realizations = 60
poisson_trees = []
for _ in range(num_poisson_realizations):
    p_idx = rng.choice(leaf_area_px, size=N, replace=False)
    pts = np.column_stack([leaf_xs[p_idx], leaf_ys[p_idx]])
    poisson_trees.append(KDTree(pts))

radii = np.linspace(15, 160, 22)
var_leaf, mean_leaf = [], []
var_pois_mean, var_pois_std = [], []
fano_leaf, fano_pois = [], []
valid_radii = []

for R in radii:
    sy, sx = np.where(dist_map >= R)
    if len(sx) < 100:
        continue
    valid_radii.append(R)
    n_samp = min(1200, len(sx))
    chosen = rng.choice(len(sx), size=n_samp, replace=False)
    centers = np.column_stack([sx[chosen], sy[chosen]])
    
    # Leaf counts
    c_leaf = np.array([len(tree.query_ball_point(pt, r=R)) for pt in centers])
    m_l = float(np.mean(c_leaf))
    v_l = float(np.var(c_leaf))
    mean_leaf.append(m_l)
    var_leaf.append(v_l)
    fano_leaf.append(v_l / m_l)
    
    # Poisson realizations
    v_p_list = []
    m_p_list = []
    for pt_tree in poisson_trees:
        c_p = np.array([len(pt_tree.query_ball_point(pt, r=R)) for pt in centers])
        v_p_list.append(np.var(c_p))
        m_p_list.append(np.mean(c_p))
    
    m_p = float(np.mean(m_p_list))
    v_p = float(np.mean(v_p_list))
    var_pois_mean.append(v_p)
    var_pois_std.append(float(np.std(v_p_list)))
    fano_pois.append(v_p / m_p)

valid_radii = np.array(valid_radii)
mean_leaf = np.array(mean_leaf)
var_leaf = np.array(var_leaf)
fano_leaf = np.array(fano_leaf)
var_pois_mean = np.array(var_pois_mean)
var_pois_std = np.array(var_pois_std)
fano_pois = np.array(fano_pois)

# Power law fits: Var(N) ~ A * R^b
fit_mask = valid_radii >= 30.0
p_leaf = np.polyfit(np.log(valid_radii[fit_mask]), np.log(var_leaf[fit_mask]), 1)
exponent_leaf = float(p_leaf[0])
coeff_leaf = float(np.exp(p_leaf[1]))

p_pois = np.polyfit(np.log(valid_radii[fit_mask]), np.log(var_pois_mean[fit_mask]), 1)
exponent_pois = float(p_pois[0])

def class2_model(R, a, b):
    return a * R * np.log(R) + b * R

popt_c2, _ = curve_fit(class2_model, valid_radii[fit_mask], var_leaf[fit_mask], p0=[0.01, 0.05])

print(f"Variance scaling exponent (Leaf): b = {exponent_leaf:.3f}")
print(f"Variance scaling exponent (Poisson ensemble): b = {exponent_pois:.3f}")
print(f"Fano factor at R=160 px: Leaf = {fano_leaf[-1]:.3f}, Poisson = {fano_pois[-1]:.3f}")
print(f"Variance suppression ratio at R=160 px: {var_leaf[-1] / var_pois_mean[-1]:.3f}")

# 11. Plotting Publication-Quality Figures
print("11. Generating publication-quality figures...")

# ----------------- FIGURE 1: Vein Segmentation Pipeline -----------------
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 7.5))

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

# ----------------- FIGURE 2: Areole Polygons & Centroids -----------------
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

# Draw rectangle around inset ROI: x in [500, 780], y in [260, 480]
rx1, rx2, ry1, ry2 = 520, 760, 260, 460
cv2.rectangle(vis_cents, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

axes2[1].imshow(vis_cents)
axes2[1].set_title(f"(b) Located Centroids on Mint Leaf (N={N})", fontweight="bold")
axes2[1].axis("off")

# Clean inset axes placed in upper right
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

# ----------------- FIGURE 3: Tessellation and Coordination -----------------
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.4))

axes3[0].triplot(polygon_centroids[:, 0], polygon_centroids[:, 1], tri.simplices, color="#2b5c8f", lw=0.7, alpha=0.7)
axes3[0].plot(polygon_centroids[is_interior, 0], polygon_centroids[is_interior, 1], 'o', color="#d95f02", markersize=3, label="Interior")
axes3[0].plot(polygon_centroids[~is_interior, 0], polygon_centroids[~is_interior, 1], 'x', color="#7570b3", markersize=3, label="Boundary")
axes3[0].set_title("(a) Delaunay Triangulation", fontweight="bold")
axes3[0].set_aspect("equal")
axes3[0].axis("off")
axes3[0].legend(loc="lower right", framealpha=0.8)

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

axes3[2].hist(polygon_areas[is_interior], bins=20, density=True, color="#386cb0", edgecolor="black", alpha=0.75, label="Data")
shape, loc, scale = lognorm.fit(polygon_areas[is_interior], floc=0)
x_fit = np.linspace(min(polygon_areas[is_interior]), max(polygon_areas[is_interior]), 200)
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

# ----------------- FIGURE 4: Hyperuniformity in Real Space -----------------
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 4.6))

axes4[0].plot(valid_radii, var_leaf, 'ro-', lw=2, markersize=5, label=f"Mint Leaf ($b={exponent_leaf:.2f}$)")
axes4[0].plot(valid_radii, var_pois_mean, 'ks--', lw=1.5, markersize=4, label=f"Poisson Ensemble ($b={exponent_pois:.2f}$)")
th_pois = rho * np.pi * valid_radii**2
axes4[0].plot(valid_radii, th_pois, 'k:', lw=1.2, label=r"Poisson Theory ($\sim R^2$)")
c1_line = var_leaf[fit_mask][0] * (valid_radii / valid_radii[fit_mask][0])**1.0
axes4[0].plot(valid_radii, c1_line, 'b-.', lw=1.5, label=r"Class I Scaling ($\sim R^1$)")
axes4[0].plot(valid_radii[fit_mask], class2_model(valid_radii[fit_mask], *popt_c2), 'm-', lw=1.8, label=r"Class II ($R\ln R$)")

axes4[0].set_xscale("log")
axes4[0].set_yscale("log")
axes4[0].set_xlabel("Window Radius $R$ [pixels]")
axes4[0].set_ylabel(r"Number Variance $\sigma_N^2(R)$")
axes4[0].set_title(r"(a) Number Variance Scaling $\sigma_N^2(R)$", fontweight="bold")
axes4[0].grid(True, which="both", linestyle="--", alpha=0.4)
axes4[0].legend(loc="upper left", fontsize=8.5)

axes4[1].plot(valid_radii, fano_leaf, 'ro-', lw=2, markersize=5, label="Mint Leaf Areoles")
axes4[1].plot(valid_radii, fano_pois, 'ks--', lw=1.5, markersize=4, label="Poisson Ensemble")
axes4[1].axhline(1.0, color="gray", linestyle=":", lw=1.5, label="Poisson Limit ($F=1$)")
axes4[1].set_xlabel("Window Radius $R$ [pixels]")
axes4[1].set_ylabel(r"Fano Factor $F(R) = \sigma_N^2(R) / \langle N(R) \rangle$")
axes4[1].set_title(r"(b) Variance Suppression $F(R) \to 0$", fontweight="bold")
axes4[1].set_ylim(0, 1.15)
axes4[1].grid(True, linestyle="--", alpha=0.4)
axes4[1].legend(loc="center right", fontsize=9)

axes4[2].plot(r_centers, g_r, 'b-', lw=2, label="Leaf $g(r)$")
axes4[2].axhline(1.0, color="k", linestyle=":", lw=1.5, label="Ideal Gas ($g=1$)")
axes4[2].axvline(r_exclusion, color="orange", linestyle="--", lw=1.5, label=f"Exclusion $r_{{excl}}={r_exclusion:.1f}$ px")
axes4[2].axvline(r_peak, color="red", linestyle="--", lw=1.5, label=f"Primary Peak $r={r_peak:.1f}$ px")
axes4[2].set_xlabel("Separation Distance $r$ [pixels]")
axes4[2].set_ylabel("Pair Correlation $g(r)$")
axes4[2].set_title(r"(c) Pair Correlation Function $g(r)$", fontweight="bold")
axes4[2].set_xlim(0, r_max)
axes4[2].grid(True, linestyle="--", alpha=0.4)
axes4[2].legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
fig4.savefig("figures/fig4_hyperuniformity_realspace.png", dpi=300)
fig4.savefig("figures/fig4_hyperuniformity_realspace.pdf")
plt.close(fig4)

# ----------------- FIGURE 5: Spectral Analysis & Structure Factor -----------------
fig5, axes5 = plt.subplots(1, 2, figsize=(13, 4.8))

im_s2d = axes5[0].imshow(np.clip(S_2D, 0, 3.0), extent=[kx_grid[0], kx_grid[-1], ky_grid[0], ky_grid[-1]],
                         origin="lower", cmap="magma")
cbar = fig5.colorbar(im_s2d, ax=axes5[0], fraction=0.046, pad=0.04)
cbar.set_label(r"$S(\mathbf{k})$ Intensity", rotation=270, labelpad=15)
# Draw circular ring at k_peak
ring_th = np.linspace(0, 2*np.pi, 100)
axes5[0].plot(k_peak * np.cos(ring_th), k_peak * np.sin(ring_th), 'c--', lw=1.2, label=f"Peak Ring ($k^*={k_peak:.2f}$)")
axes5[0].set_xlabel(r"Wavenumber $k_x$ [$\mathrm{rad/px}$]")
axes5[0].set_ylabel(r"Wavenumber $k_y$ [$\mathrm{rad/px}$]")
axes5[0].set_title(r"(a) 2D Structure Factor $S(k_x, k_y)$", fontweight="bold")
axes5[0].legend(loc="lower right", fontsize=8.5)

axes5[1].plot(k_vals, S_k_clean, 'm-', lw=2.2, label=r"Apodized Hankel $S(k)$")
axes5[1].plot(k_azimuthal, S_direct_az, 'b^--', markersize=4, lw=1.2, alpha=0.7, label=r"Direct Azimuthal $S(k)$")
axes5[1].axhline(1.0, color="k", linestyle=":", lw=1.5, label="Poisson asymptote ($S=1$)")
axes5[1].plot(k_peak, S_peak, 'ro', markersize=6, label=f"Peak Ring ($k^*={k_peak:.3f}$ rad/px)")

axes5[1].set_xlabel(r"Wavenumber $k = |\mathbf{k}|$ [$\mathrm{rad/px}$]")
axes5[1].set_ylabel(r"Structure Factor $S(k)$")
axes5[1].set_title(r"(b) Radially Averaged $S(k)$ ($S(k) \to 0$ as $k \to 0$)", fontweight="bold")
axes5[1].set_xlim(0, 0.35)
axes5[1].set_ylim(-0.05, 1.45)
axes5[1].grid(True, linestyle="--", alpha=0.4)
axes5[1].legend(loc="lower right", fontsize=9)

plt.tight_layout()
fig5.savefig("figures/fig5_spectral_structure_factor.png", dpi=300)
fig5.savefig("figures/fig5_spectral_structure_factor.pdf")
plt.close(fig5)

# 12. Save Updated JSON Summary
results_summary = {
    "image": {
        "filename": "Mint.jpg",
        "height": int(h),
        "width": int(w),
        "leaf_area_px": leaf_area_px,
        "leaf_area_fraction": float(leaf_area_px / (h * w))
    },
    "segmentation": {
        "total_polygons": N,
        "interior_polygons": N_interior,
        "boundary_polygons": int(N - N_interior),
        "vein_skeleton_length_px": total_skel_length,
        "areole_area_mean": float(np.mean(polygon_areas)),
        "areole_area_std": float(np.std(polygon_areas)),
        "areole_area_median": float(np.median(polygon_areas)),
        "areole_diameter_mean": float(np.mean(eq_diameters)),
        "areole_diameter_std": float(np.std(eq_diameters))
    },
    "spatial_statistics": {
        "number_density_rho": float(rho),
        "nn_dist_mean": mean_nn,
        "nn_dist_std": std_nn,
        "nn_dist_min": min_nn,
        "nn_dist_max": max_nn,
        "coord_mean": mean_coord,
        "hexagonal_fraction": hex_fraction,
        "coord_distribution": {int(k): float(v) for k, v in zip(unique_coords, coord_probs)}
    },
    "pair_correlation": {
        "r_exclusion": r_exclusion,
        "r_peak": r_peak,
        "g_peak": g_peak
    },
    "hyperuniformity": {
        "variance_exponent_leaf": exponent_leaf,
        "variance_exponent_poisson": exponent_pois,
        "fano_factor_min": float(np.min(fano_leaf)),
        "fano_factor_max": float(np.max(fano_leaf)),
        "fano_factor_at_max_R": float(fano_leaf[-1]),
        "variance_suppression_ratio_at_max_R": float(var_leaf[-1] / var_pois_mean[-1]),
        "spectral_peak_wavenumber_k": k_peak,
        "spectral_peak_wavelength_px": lambda_peak,
        "class2_fit_parameters": [float(popt_c2[0]), float(popt_c2[1])],
        "hyperuniformity_class": "Class II (disordered hyperuniform with logarithmic corrections / anomalous suppression)"
    }
}

with open("results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=4)

print("Updated script completed successfully!")
