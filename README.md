# Disordered Hyperuniformity in Biological Networks
### A Hands-On Computational Laboratory with Mint Leaf Venation (*Mentha spicata*)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Pedagogical Disclaimer:**  
> This repository provides an illustrative, hands-on computational tutorial for courses in **Hyperuniformity, Statistical Mechanics, and Biological Physics**. It demonstrates the end-to-end practical workflow required to extract point patterns from experimental biological images, perform Voronoi/Delaunay tessellations, compute real-space number variance scaling $\sigma_N^2(R)$, evaluate Fano factor suppression $F(R)$, and determine reciprocal-space static structure factors $S(\mathbf{k})$.  
> **This is an instructional case study, NOT a mathematical proof of asymptotic thermodynamic hyperuniformity in plant leaves**, which are fundamentally bounded by finite-size boundaries, developmental gradients, and physiological variations.

---

## Table of Contents
1. [Physical & Biological Background](#physical--biological-background)
2. [Mathematical Foundations of Hyperuniformity](#mathematical-foundations-of-hyperuniformity)
3. [The Computational Pipeline](#the-computational-pipeline)
4. [Key Quantitative Findings](#key-quantitative-findings)
5. [Repository Structure](#repository-structure)
6. [Quick Start & Reproducibility](#quick-start--reproducibility)
7. [Hands-On Course Exercises](#hands-on-course-exercises)
8. [References & Further Reading](#references--further-reading)

---

## Physical & Biological Background

In dicotyledonous angiosperms, leaf venation networks solve a multi-objective evolutionary optimization problem:
1. **Hydraulic Transport:** Water exiting xylem vessels must diffuse through high-resistance mesophyll to evaporating stomatal sites. Minor veins must be distributed uniformly to prevent localized desiccation.
2. **Damage Resilience:** A hierarchical reticulated network with closed loops provides redundant bypass pathways around insect lacerations or air embolisms (cavitation).
3. **Mechanical Compliance:** Veins act as structural struts, balancing flexural rigidity against tearing in turbulent wind.

The inter-vein photosynthetic compartments are known as **areoles**. By identifying the geometric centroids of these areoles, we obtain an experimental two-dimensional point pattern. In statistical physics, such disordered cellular networks raise a fundamental question: **do the areole centers organize into a disordered hyperuniform state?**

---

## Mathematical Foundations of Hyperuniformity

### 1. Local Number Variance $\sigma_N^2(R)$
Consider a 2D point pattern of $N$ centroids $\{\mathbf{r}_j\}_{j=1}^N$ with mean number density $\rho = N / A$. Inside a circular observation window of radius $R$ centered at $\mathbf{x}_0$, the number of enclosed points is:
$$N(R; \mathbf{x}_0) = \sum_{j=1}^N \Theta(R - \|\mathbf{r}_j - \mathbf{x}_0\|)$$
where $\Theta$ is the Heaviside step function. The expected count is $\langle N(R) \rangle = \rho \pi R^2$. The local number variance is:
$$\sigma_N^2(R) \equiv \langle [N(R) - \langle N(R) \rangle]^2 \rangle$$

* **Poisson Baseline (Complete Spatial Randomness, CSR):**  
  Variance equals the mean, scaling with window area:
  $$\sigma_{N,\mathrm{Poisson}}^2(R) = \langle N(R) \rangle \sim R^2 \implies F(R) \equiv \frac{\sigma_N^2(R)}{\langle N(R) \rangle} = 1$$
* **Hyperuniformity Criterion (Torquato & Stillinger, 2003):**  
  A point pattern is hyperuniform if the variance grows asymptotically slower than the window volume:
  $$\lim_{R \to \infty} \frac{\sigma_N^2(R)}{\pi R^2} = 0 \iff \lim_{R \to \infty} F(R) = 0$$

### 2. Torquato-Stillinger Scaling Classes (in 2D)
* **Class I ($p = 1$):** Surface-area scaling $\sigma_N^2(R) \sim R$ (periodic crystals, quasicrystals, stealthy ground states).
* **Class II ($p = 1$ with logarithmic correction):** $\sigma_N^2(R) \sim R \ln R$ (maximally random jammed sphere packings, critical cellular foams).
* **Class III ($1 < p < 2$):** Power-law suppression $\sigma_N^2(R) \sim R^p$ with $p < 2$ (disordered biological mosaics, defected lattices).

### 3. Reciprocal-Space Static Structure Factor $S(\mathbf{k})$
For wavevector $\mathbf{k} \neq \mathbf{0}$:
$$S(\mathbf{k}) \equiv \frac{1}{N} \left| \sum_{j=1}^N e^{-i \mathbf{k} \cdot \mathbf{r}_j} \right|^2$$
Hyperuniformity requires the structure factor to vanish at zero wavenumber:
$$\lim_{|\mathbf{k}| \to 0} S(\mathbf{k}) = 0$$

### 4. Real-Space Pair Correlation $g(r)$ & Hankel Transform
For an isotropic 2D system, $S(k)$ is related to the radial distribution function $g(r)$ via a Fourier-Bessel (Hankel) transform:
$$S(k) = 1 + \rho \int_0^\infty 2\pi r [g(r) - 1] J_0(kr) \, dr$$
where $J_0$ is the zeroth-order Bessel function of the first kind.

---

## The Computational Pipeline

The accompanying script [`generate_analysis.py`](./generate_analysis.py) executes a fully automated 11-step pipeline:

```
 Mint.jpg (Raw Image)
        │
        ▼
 [1] CIELAB Color Space Filtering (a* < 120 separates leaf from background)
        │
        ▼
 [2] Multi-scale Black Top-Hat Transform + Sobel Gradient (Elevation Surface)
        │
        ▼
 [3] Marker-Controlled Watershed Segmentation (Watershed Basins = Areoles)
        │
        ▼
 [4] Medial Axis Skeletonization & Geometric Centroid Extraction
        │
        ▼
 [5] Delaunay Triangulation & Coordination Distribution P(z)
        │
        ▼
 [6] Unbiased Boundary-Eroded Window Sampling for sigma_N^2(R) & Fano Factor
        │
        ▼
 [7] Pair Correlation Function g(r) & Static Structure Factor S(k)
        │
        ▼
 Publication Figures (fig1 - fig5) + JSON Summary + LaTeX Report
```

1. **Color-Space Separation:** Leaf lamina is isolated from the background using chrominance $a^* < 120$ in CIELAB color space, followed by morphological hole filling and Euclidean distance transform.
2. **Top-Hat Morphological Enhancement:** Minor veins appear as dark grooves on the illuminated lamina. A multi-scale Black Top-Hat (BTH) transform using circular structuring elements ($k \in \{7, 11, 15, 21\}\text{ px}$) extracts dark vascular channels regardless of varying background illumination.
3. **Watershed Partitioning:** The elevation surface $H(\mathbf{x}) = \mathrm{BTH}(\mathbf{x}) + 0.5 \|\nabla I(\mathbf{x})\|$ is flooded from detected regional minima to obtain watertight polygonal areoles.
4. **Centroid Localization & Filtering:** True geometric centroids $\mathbf{C}_i = (\bar{x}_i, \bar{y}_i)$ are calculated. Cells intersecting the outer boundary are classified as boundary tiles to avoid edge truncation artifacts.
5. **Delaunay Dual Graph:** Delaunay triangulation of the interior centroids gives the topological coordination number $z_i$.
6. **Boundary-Eroded Real-Space Sampling:** Windows crossing the leaf perimeter artificially alter variance scaling. Sampling window centers $\mathbf{x}_0$ are strictly confined to the interior domain $\Omega_R = \{\mathbf{x} : D(\mathbf{x}) \ge R\}$, evaluated against an ensemble of 60 synthetic Poisson point processes of identical density and geometry.
7. **Reciprocal-Space Analysis:** $S(\mathbf{k})$ is computed via window-apodized Hankel transformation of $g(r)$ and direct 2D discrete Fourier summation.

---

## Key Quantitative Findings

| Metric | Spearmint Leaf (*Mentha spicata*) | Poisson Benchmark (CSR) | Biophysical / Physical Significance |
| :--- | :---: | :---: | :--- |
| **Analyzed Lamina Area** | $624,794\text{ px}^2$ ($55.18\%$ of frame) | Identical | Total photosynthetic domain |
| **Total Areoles ($N$)** | $288$ ($237$ interior) | $288$ points | Mesophyll compartments |
| **Vein Skeleton Length** | $25,628\text{ px}$ | -- | Medial-axis minor vein network |
| **Mean Areole Area $\langle A \rangle$** | $2169.4 \pm 1347.8\text{ px}^2$ | -- | Lognormally distributed ($R^2 = 0.94$) |
| **Mean Coordination $\langle z \rangle$** | $\mathbf{5.90 \pm 1.05}$ | $\approx 6$ | Strictly obeys Euler trivalent foam law ($\langle z \rangle \approx 6$) |
| **Hexagonal Tiles $P(6)$** | $\mathbf{32.9\%}$ | $29.8\%$ | Peak coordination with balanced $5$--$7$ disclinations |
| **Nearest-Neighbor Distance** | $35.83 \pm 8.57\text{ px}$ | $23.28 \pm 11.95\text{ px}$ | Steric exclusion narrows spacing variance |
| **Pair Correlation Hole $r_{\mathrm{excl}}$** | $\mathbf{14.0\text{ px}}$ | $0.0\text{ px}$ | Strict hard-core exclusion zone ($g(r) = 0$) |
| **Primary Peak $r_{\mathrm{peak}}$** | $\mathbf{46.0\text{ px}}$ ($g_{\max} = 1.34$) | $g(r) \equiv 1$ | Short-range liquid-like coordination shell |
| **Variance Exponent ($b$)** | $\mathbf{1.642}$ ($\sigma_N^2 \sim R^{1.64}$) | $\mathbf{1.833}$ | **Sub-Poissonian scaling ($b < 2$)** |
| **Fano Factor at $R = 160\text{ px}$** | $\mathbf{0.170}$ | $0.773$ | **Anomalous variance suppression** |
| **Variance Ratio vs. Poisson** | $\mathbf{0.223}$ ($77.7\%$ suppression) | $1.000$ | Large-scale density fluctuation suppression |
| **Structure Factor Low-$k$ Basin** | $S(k \to 0) \to 0.20$ | $S(k) \equiv 1$ | Pronounced depletion basin |
| **Bragg-like Liquid Peak** | $k^* = 0.144\text{ rad/px}$ ($\lambda^* = 43.6\text{ px}$) | Flat | Mean inter-areole characteristic spacing |
| **Hyperuniformity Regime** | **Class II / III Disordered HU** | Non-hyperuniform | Disordered hyperuniform cellular network |

---

## Repository Structure

```
Mint/
├── Mint.jpg                               # High-resolution input macro photograph
├── generate_analysis.py                   # Self-contained processing & figure generation script
├── results_summary.json                   # Machine-readable output metrics
├── mint_leaf_hyperuniformity_report.tex   # Full academic LaTeX report (8 pages)
├── mint_leaf_hyperuniformity_report.pdf   # Compiled publication-ready PDF report
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Clean git configuration
├── figures/                               # Publication-quality vector & raster figures
│   ├── fig1_vein_segmentation.pdf / .png        # Image segmentation & skeletonization
│   ├── fig2_areole_polygons.pdf / .png          # Watertight polygons & centroids
│   ├── fig3_tessellation_coordination.pdf / .png# Delaunay graph, P(z), and area distribution
│   ├── fig4_hyperuniformity_realspace.pdf / .png# sigma_N^2(R), Fano factor F(R), and g(r)
│   └── fig5_spectral_structure_factor.pdf / .png# 2D S(kx, ky) map and radial Hankel S(k)
└── README.md                              # This pedagogical laboratory guide
```

---

## Quick Start & Reproducibility

### Prerequisites
* Python 3.10 or higher
* (Optional) TeX Live / `pdflatex` to recompile the PDF report

### 1. Clone & Set Up Environment
```bash
git clone git@github.com:elomba/Mint.git
cd Mint

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required scientific packages
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline
```bash
python generate_analysis.py
```
This script will:
* Segment the leaf lamina and veins from `Mint.jpg`.
* Extract all 288 areole polygons and locate their centroids.
* Compute Delaunay triangulation and topological statistics.
* Perform real-space window sampling across radii $R \in [15, 160]\text{ px}$ with 60 Poisson realizations.
* Calculate the pair correlation $g(r)$ and Hankel/Fourier structure factors $S(\mathbf{k})$.
* Save `results_summary.json` and generate all 5 figures in the `figures/` directory (both 300 DPI `.png` and vector `.pdf`).

### 3. (Optional) Recompile the LaTeX Report
```bash
pdflatex -interaction=nonstopmode mint_leaf_hyperuniformity_report.tex
pdflatex -interaction=nonstopmode mint_leaf_hyperuniformity_report.tex
```

---

## Hands-On Course Exercises

For students taking a course on **Hyperuniformity and Complex Networks**, here are suggested assignments:

1. **Structuring Element Sensitivity Analysis:**  
   In `generate_analysis.py`, modify the morphological filter scales:
   ```python
   for k in [7, 11, 15, 21]: # Try finer [3, 7, 11] or coarser [15, 25, 35]
   ```
   How does over-segmentation or under-segmentation affect the measured number variance scaling exponent $b$?
2. **Window Geometry: Disks vs. Squares:**  
   Replace circular observation disks with square windows of side length $L = 2R$. Verify whether the variance scaling exponent $b$ is invariant to window geometry, as predicted by Torquato (2018).
3. **Random Shuffle Null Model:**  
   Implement a randomized control: perturb each centroid $\mathbf{C}_i \to \mathbf{C}_i + \boldsymbol{\delta}_i$, where $\boldsymbol{\delta}_i$ is a Gaussian random displacement with variance $\sigma^2$. At what displacement amplitude does the hyperuniform Fano factor suppression $F(R) \to 0$ transition back to Poissonian behavior $F(R) \approx 1$?
4. **Boundary Truncation Effects:**  
   Intentionally sample windows that cross the leaf perimeter $\partial \Omega$ without distance-transform erosion. Observe how edge crossing introduces an artificial surface term $\sigma_N^2 \sim R$ that can mimic Class I hyperuniformity.

---

## References & Further Reading

1. **Torquato, S. & Stillinger, F. H.** (2003). *Local density fluctuations, hyperuniformity, and order metrics.* Physical Review E, 68(4), 041113. [DOI:10.1103/PhysRevE.68.041113](https://doi.org/10.1103/PhysRevE.68.041113)
2. **Torquato, S.** (2018). *Hyperuniform states of matter.* Physics Reports, 745, 1–95. [DOI:10.1016/j.physrep.2018.03.001](https://doi.org/10.1016/j.physrep.2018.03.001)
3. **Katifori, E., Szöllősi, G. J., & Magnasco, M. O.** (2010). *Damage and fluctuations induce loops in optimal transport networks.* Physical Review Letters, 104(4), 048704. [DOI:10.1103/PhysRevLett.104.048704](https://doi.org/10.1103/PhysRevLett.104.048704)
4. **Sack, L. & Scoffoni, C.** (2013). *Leaf venation: structure, function, development, evolution, ecology and applications in the genomics era.* New Phytologist, 198(4), 983–1000. [DOI:10.1111/nph.12253](https://doi.org/10.1111/nph.12253)
5. **Jiao, Y., Lau, T., Hatzikirou, H., Meyer-Hermann, M., Corbo, J. C., & Torquato, S.** (2014). *Avian photoreceptor patterns represent a disordered hyperuniform solution to a multiscale packing problem.* Physical Review E, 89(2), 022721. [DOI:10.1103/PhysRevE.89.022721](https://doi.org/10.1103/PhysRevE.89.022721)
6. **Corson, F., Adda-Bedia, M., & Boudaoud, A.** (2009). *Turning a leaf: from mechanical deformation to visual aesthetics of leaf venation.* Physical Review Letters, 103(25), 258101. [DOI:10.1103/PhysRevLett.103.258101](https://doi.org/10.1103/PhysRevLett.103.258101)
7. **Klatt, M. A., Lovrić, J., Chen, D., Kapfer, S. C., Schaller, F. M., Schönhöfer, P. W., ... & Torquato, S.** (2019). *Universal properties of disordered hyperuniform cellular networks.* Nature Communications, 10(1), 811. [DOI:10.1038/s41467-019-08360-5](https://doi.org/10.1038/s41467-019-08360-5)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
