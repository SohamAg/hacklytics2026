# HeartScape

**3D cardiac morphology explorer for congenital heart disease structural analysis.**

HeartScape takes segmented MRI scans, extracts anatomical volumes, renders interactive 3D meshes,
and places each patient in a learned morphology space alongside 60 annotated CHD patients —
enabling structural comparison, condition simulation, and vector-similarity-driven soft diagnosis.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Pipeline Overview](#pipeline-overview)
4. [Feature Extraction](#feature-extraction)
5. [Condition Analysis & Mesh Morphing](#condition-analysis--mesh-morphing)
6. [PCA Morphology Landscape](#pca-morphology-landscape)
7. [Actian VectorAI Integration](#actian-vectorai-integration)
8. [Flask Backend API](#flask-backend-api)
9. [Frontend](#frontend)
10. [Running the Project](#running-the-project)

---

## Project Structure

```
hacklytics2026/
├── data-processing/
│   ├── backend_pipeline.py       # Orchestration: analyze_scan / simulate_heart
│   ├── condition_analysis.py     # Reference, condition effects, multipliers
│   ├── feature_processing.py     # PCA training + scaler fitting
│   ├── morphology_engine.py      # PCA projection, similarity search, severity
│   ├── mesh_rendering.py         # NIfTI → OBJ export (PyVista marching cubes)
│   ├── vectorai_client.py        # Actian VectorAI multi-resolution search client
│   ├── vectorai_setup.py         # One-time ingestion into VectorAI DB
│   ├── server.py                 # Flask API server
│   ├── heart_features.csv        # Extracted volumetric features for 60 patients
│   └── hvsmr_clinical.csv        # Ground-truth diagnosis labels per patient
├── models/
│   ├── pca_model.pkl             # Trained sklearn PCA (2 components)
│   ├── scaler_model.pkl          # Trained StandardScaler
│   ├── pca_dataset.csv           # Training set PC1/PC2 projections
│   └── vectorai_scaler.json      # Z-score parameters for VectorAI queries
├── index.html                    # Frontend entry point
├── heart-scene.js                # Three.js scene, mesh morphing, UI logic
└── lens-module.js                # AI selection overlay (Gemini integration)
```

---

## Dataset

**HVSMR-2.0** — a publicly available benchmark of 60 pediatric cardiac MRI scans from
Children's Hospital Boston. Each scan is manually segmented into 8 anatomical structures:

| Label | Structure |
|-------|-----------|
| 1 | Left Ventricle (LV) |
| 2 | Right Ventricle (RV) |
| 3 | Left Atrium (LA) |
| 4 | Right Atrium (RA) |
| 5 | Aorta (AO) |
| 6 | Pulmonary Artery (PA) |
| 7 | Superior Vena Cava (SVC) |
| 8 | Inferior Vena Cava (IVC) |

Conditions span 15+ CHD categories including VSD, ASD, DORV, D-Loop TGA, L-Loop TGA,
Arterial Switch, Atrial Switch, Single Ventricle, Fontan, Glenn, Heterotaxy, Dextrocardia,
Mild/Moderate Dilation, and Normal.

**Holdout patients:** `[1, 14, 16, 28, 57]` are excluded from all training (PCA, VectorAI
ingestion) and used as genuinely unseen demo inputs.

---

## Pipeline Overview

```
.nii.gz segmentation
        │
        ▼
 Feature Extraction          ← voxel label counts × voxel spacing → mL
        │
        ├──► Mesh Rendering  ← marching cubes per label → decimated OBJ
        │
        ├──► PCA Projection  ← StandardScaler → PCA(2) → (PC1, PC2)
        │
        ├──► Similarity      ← Euclidean distance in PCA space → k=3 nearest
        │
        ├──► Severity Score  ← PC1 normalized to [0, 1] across training range
        │
        └──► VectorAI Search ← z-score vec → 6 collections → soft attribution
```

---

## Feature Extraction

From `backend_pipeline.py / extract_features_from_nii`:

```python
seg  = nib.load(nii_path)
data = seg.get_fdata()
vox  = np.prod(seg.header.get_zooms())    # mm³ per voxel

vol_mm3 = np.sum(data == label_id) * vox
vol_ml  = vol_mm3 / 1000.0
```

Real voxel spacing from the NIfTI affine header is used so volumes are physically meaningful
in millilitres — not voxel counts. Four derived ratios are also computed:

```
LV_RV_ratio  = LV / RV
LA_RA_ratio  = LA / RA
AO_fraction  = Aorta / Total_heart_vol
LV_fraction  = LV / Total_heart_vol
```

The 9-dimensional feature vector used by PCA and VectorAI is:

```
x = [Label_1_vol_ml, ..., Label_8_vol_ml, Total_heart_vol]
```

---

## Condition Analysis & Mesh Morphing

### Reference vector

The normal cohort (patients flagged `Normal=1` with no other defects) defines a reference:

```
ref[f] = mean({ x[f] : patient is Normal })
```

### Per-condition effects

For each condition `c` with ≥ 2 patients:

```
effect[c][f] = mean({ x[f] : patient has condition c })
```

### Multiplicative scaling

The volume multiplier for condition `c` on feature `f`:

```
m[c][f] = clip( effect[c][f] / ref[f], 0.3, 3.0 )
```

### Multi-condition blending

When multiple conditions are selected, multipliers are combined via **geometric mean**:

```
m_blend[f] = exp( mean({ log(m[c][f]) : c ∈ selected }) )
estimated[f] = ref[f] × m_blend[f]
```

This preserves multiplicative symmetry — blending a 2× and 0.5× effect yields 1× (neutral),
which is correct under log-space averaging.

### Mesh scale factor

To scale a 3D mesh proportionally to a volume change, the linear dimension factor is:

```
scale[f] = clip(estimated[f] / ref[f], 0.2, 5.0) ^ (1/3)
```

The cube root maps a volume ratio to a linear scale ratio so the mesh visually represents
the correct volumetric change.

---

## PCA Morphology Landscape

From `feature_processing.py` and `morphology_engine.py`.

### Training

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)    # zero mean, unit variance per feature

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

Fitted on 55 patients (holdouts excluded). Both models serialized to `models/`.

### Projection (inference)

```python
vec_scaled = scaler.transform(x.reshape(1, -1))
(pc1, pc2) = pca.transform(vec_scaled)[0]
```

### Interpretation

- **PC1** (dominant component) correlates with overall heart size and total chamber volume —
  larger, more dilated hearts project to higher PC1
- **PC2** captures shape variation — chamber-to-vessel balance and ventricular dominance patterns

### PCA similarity search

Nearest training patients in 2D PCA space:

```
distance_i = || (pc1, pc2) - (pc1_i, pc2_i) ||_2
```

Top-3 by ascending distance are returned with their ground-truth category labels.

### Severity score

PC1 is normalized across the training range to produce a [0, 1] severity proxy:

```
severity = clip( (pc1 - pc1_min) / (pc1_max - pc1_min), 0, 1 )
```

---

## Actian VectorAI Integration

### Why vector search

A hard classifier assigns a single label per patient. CHD anatomy is a continuous spectrum —
a patient may have chambers resembling one condition and vessels resembling another. Vector
similarity search returns ranked, distance-weighted matches across all known anatomies without
requiring a training label for every possible morphology combination.

### Collections (6 total)

| Collection | Dim | Contents |
|---|---|---|
| `heart_patients` | 9D | One vector per non-holdout patient |
| `condition_archetypes` | 9D | Mean anatomy per condition group (whole-heart) |
| `heart_chambers` | 4D | LV, RV, LA, RA only |
| `heart_vessels` | 4D | Aorta, PA, SVC, IVC only |
| `chamber_archetypes` | 4D | Chamber-only archetype per condition |
| `vessel_archetypes` | 4D | Vessel-only archetype per condition |

All collections use `DistanceMetric.EUCLIDEAN`.

### Z-score standardization

Raw features are standardized before ingestion and before every query:

```python
z[f] = (x[f] - mean_train[f]) / std_train[f]
```

This is critical. Cosine similarity only measures vector direction — a heart with every chamber
2× normal size has near-identical cosine similarity to a normal heart because the ratios are
preserved. Our first implementation returned `0.999` similarity for every patient for exactly
this reason. Euclidean distance in z-score space correctly captures:
**how many population standard deviations away is this patient, per feature.**

Scaler parameters are saved to `models/vectorai_scaler.json` and loaded by the client at
query time to standardize incoming feature vectors into the same space as stored vectors.

### Multi-resolution query

Every patient upload runs all six searches in parallel:

```python
full_patients    = search("heart_patients",       z_full,     k=5)
chamber_patients = search("heart_chambers",       z_chambers, k=5)
vessel_patients  = search("heart_vessels",        z_vessels,  k=5)
full_archetypes  = search("condition_archetypes", z_full,     k=5)
chamber_archs    = search("chamber_archetypes",   z_chambers, k=5)
vessel_archs     = search("vessel_archetypes",    z_vessels,  k=5)
```

This surfaces cases where a patient's overall anatomy is unremarkable but their vessel geometry
is anomalous — a signal that collapses in a single 9D search.

### Soft condition attribution

Archetype distances are converted to probability-like scores via **softmax over negative
Euclidean distances** with an adaptive temperature:

```python
spread = d_max - d_min
temp   = min(3.0 / max(spread, 0.05), 6.0)   # auto-scaled, capped

scores = exp(-d * temp)
scores = scores / scores.sum()                 # normalize to sum = 1
```

**Why adaptive temperature:**
- Small spread (all conditions equally distant, genuinely ambiguous): `temp` stays low,
  scores stay diffuse
- Large spread (one clear match): `temp` is higher, the top condition stands out proportionally
- The cap at `6.0` prevents a single condition from dominating completely when the spread is tiny

A naive `1/(1+d)` similarity formula compresses all scores into a narrow `~40%` band when
distances are similar — the softmax with temperature is what makes the attribution actually
discriminative.

### Anomaly scoring

The Euclidean distance from the nearest patient in z-score space maps to a severity label:

```
d < 1.0  → within normal range      (< 1 joint std-dev from nearest known case)
d 1–2    → mild deviation
d 2–3    → moderate anomaly
d 3+     → significant anomaly
```

### Infrastructure

VectorAI runs as a Docker container on `localhost:50051` via gRPC:

```bash
docker compose up -d          # start VectorAI
python vectorai_setup.py      # one-time ingestion
```

The server checks liveness on startup and degrades gracefully — if the container is not
running, condition attribution falls back to the same softmax temperature logic applied
directly against precomputed condition centroids.

---

## Flask Backend API

Run from `data-processing/`:

```bash
pip install flask flask-cors
python server.py
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Liveness check, reports VectorAI status |
| `POST` | `/api/simulate` | `{ "conditions": ["VSD", "ASD"] }` → analysis JSON |
| `POST` | `/api/upload` | Multipart `.nii.gz` file → analysis JSON + OBJ URL |
| `GET` | `/scan_outputs/<file>` | Serves generated OBJ meshes |

All endpoints return the same analysis payload shape:

```json
{
  "patient_id": 14,
  "ground_truth": "DORV",
  "features": { "Label_1_vol_ml": 42.3, "..." : "..." },
  "mesh_scales": { "LV": 1.12, "RV": 0.94, "..." : "..." },
  "pca": { "pc1": -1.24, "pc2": 0.38 },
  "severity": 0.31,
  "nearest": [ { "patient_id": 7, "category": "DORV", "distance": 0.42 } ],
  "condition_scores": { "DORV": 0.52, "VSD": 0.28, "..." : "..." },
  "vector_search": { "anomaly_label": "mild deviation", "..." : "..." }
}
```

---

## Frontend

Built with vanilla Three.js (no framework). Served from the project root:

```bash
python -m http.server 8080
# open http://localhost:8080
```

### Dual 3D viewport

- **Left panel (fixed reference):** Patient 1's heart, always loaded from `models/pat1.obj`.
  Color-coded per anatomical label with animated heartbeat.
- **Right panel (two modes):**

**Simulate tab** — select any CHD condition from the dropdown. The right mesh morphs in real time
using the cube-root scale factors computed from condition multipliers. Green/red outlines and
numerical deltas are displayed per structure.

**Upload tab** — drop a `.nii.gz` file or select a demo patient. The backend pipeline runs,
the generated OBJ is loaded, and the panel displays: condition attribution bars, PCA position,
anomaly label, nearest patients, and volume comparisons per structure.

### PCA scatter plot

An interactive SVG chart in the right panel shows all 60 training patients color-coded by
condition category. Clicking expands to a full modal with grid lines, axis labels, and a
complete category legend. Previously uploaded patients accumulate as amber dots across sessions.

### LensModule

A Google Lens-style selection overlay for the Three.js canvas. Box-select anatomical regions:

1. `THREE.Raycaster` picks all meshes intersecting the selection rectangle and groups them
   by anatomical name
2. Pre-computed volumes in mL from the pipeline JSON are read directly (not approximated
   from geometry) and sent to **Google Gemini** alongside bounding box dimensions and
   elongation ratios
3. A draggable popup renders the AI analysis as a multi-turn chat thread
4. "Save as Note" exports the full conversation to a sticky note on the canvas

---

## Running the Project

### Prerequisites

```bash
pip install flask flask-cors nibabel numpy pandas scikit-learn joblib pyvista scipy
```

### 1. Start VectorAI (optional but recommended)

```bash
docker compose up -d
cd data-processing
python vectorai_setup.py
```

### 2. Start the Flask backend

```bash
cd data-processing
python server.py
# → http://localhost:5000
```

### 3. Serve the frontend

```bash
# from project root
python -m http.server 8080
# → http://localhost:8080
```

### 4. Demo uploads

The following patients are held out from training and can be uploaded as `.nii.gz` demo inputs:

| Patient | Condition |
|---------|-----------|
| pat1 | Normal (reference) |
| pat14 | DORV |
| pat16 | D-Loop TGA |
| pat28 | Fontan |
| pat57 | Single Ventricle |

---

## Built With

`Python` · `Flask` · `nibabel` · `PyVista` · `scikit-learn` · `NumPy` · `pandas` ·
`Three.js` · `Actian VectorAI` · `Google Gemini API` · `Docker` · `HVSMR-2.0`
