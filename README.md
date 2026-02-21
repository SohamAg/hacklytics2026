# hacklytics2026
Hacklytics — HeartScape: 3D cardiac visualization and condition–feature analysis.

The viewer uses the provided 3D heart model (`SubTool-0-7412864.OBJ`). Run a local server (e.g. `python3 -m http.server 8000`) and open `index.html`.

## Condition–feature analysis (data-processing)

Discover how congenital conditions relate to heart chamber/vessel volumes and estimate heart shape for a chosen condition (or mix):

1. **Trend discovery** — Which conditions change LV, RV, LA, RA, etc. vs normal:
   ```bash
   cd data-processing
   pip install -r requirements.txt
   python discover_trends.py
   ```
   Prints per-condition deltas (>5% change). Use `--estimate VSD ASD` to see estimated volumes and scaling.

2. **Export for viewer** — Writes `models/condition_effects.json`. The 3D viewer uses this in the **Simulate** dropdown to scale the heart (overall size from condition data when using a single mesh model).

3. **Estimate for any condition set** — In code: `condition_analysis.estimate_features(["VSD", "ASD"], reference, condition_multipliers)` returns estimated volumes.

## PyVista heart (Trame — interactive 3D in iframe)

The **PyVista heart** page embeds a **Trame** app: your polished NIfTI heart with **full 3D interaction** (rotate, zoom, pan) in the browser. Trame is PyVista’s recommended web backend and runs as its own server (no Streamlit/stpyvista).

1. **Install** (in `data-processing/`):
   ```bash
   pip install -r requirements.txt
   ```
   Or: `pip install pyvista nibabel scikit-image "pyvista[trame]"`.
2. **Run the Trame app**:
   ```bash
   cd data-processing
   python trame_heart.py
   ```
   Optional: `python trame_heart.py cropped/cropped/pat0_cropped_seg.nii.gz`. Server runs at **http://localhost:8080**.
3. **Open the main site** (e.g. `python3 -m http.server 8000` in project root) and click **PyVista heart**. The page embeds the Trame app in an iframe; use the 3D viewer inside the iframe to rotate and zoom. If the iframe is blank, start `trame_heart.py` and refresh, or open **http://localhost:8080** in a new tab.
