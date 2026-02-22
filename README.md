# 🫀 HeartScape — Hacklytics 2026

HeartScape is an interactive 3D cardiac visualization and analytics platform built for Hacklytics 2026.  
It transforms medical imaging and condition-based data into explorable heart models, allowing users to visualize, analyze, and compare healthy and diseased anatomy in real time.

---

## 🚀 Overview

HeartScape combines:

- 🧠 **Data-driven analysis** of cardiac conditions  
- 🫀 **Interactive 3D heart visualization**  
- 📊 **Feature trend discovery from structured datasets**  
- 🔍 **Lens-style AI interaction module (Three.js + Gemini API)**  

The goal is to make cardiac education and analysis more intuitive, visual, and interactive.

---

## ✨ Features

### 1️⃣ Interactive 3D Viewer
- Built with **Three.js**
- Rotate, zoom, and inspect a 3D heart model
- Modular scene structure (`heart-scene.js`)
- OBJ-based heart model rendering

### 2️⃣ Condition–Feature Analysis
- Python-based trend extraction
- Identifies volume changes and structural deltas
- Outputs structured data for visualization integration

### 3️⃣ AI Lens Module
- Google Lens–style selection tool
- Select a portion of the 3D canvas
- Sends captured region to Gemini API
- Returns AI-powered anatomical insights

---

## 🏗️ Tech Stack

### Frontend
- HTML5
- JavaScript (ES6)
- Three.js

### AI Integration
- Google Gemini API

### Data Processing
- Python 3.8+
- NumPy / Pandas
- PyVista (optional advanced visualization)
- Trame (optional)


---

## ⚙️ Setup & Usage

### 🔹 Run the Web Viewer

1. Clone the repository:
   ```bash
   git clone https://github.com/SohamAg/hacklytics2026.git
   cd hacklytics2026
   ```

2. Start a local server:

   ```bash
   python3 -m http.server 8000
   ```

3. Open in browser:

   ```
   http://localhost:8000
   ```

---

### 🔹 Run Data Processing

1. Navigate to data-processing:

   ```bash
   cd data-processing
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run analysis:

   ```bash
   python discover_trends.py
   ```

---

### 🔹 Optional: Advanced PyVista Viewer

Install dependencies:

```bash
pip install pyvista nibabel scikit-image "pyvista[trame]"
```

Run:

```bash
python trame_heart.py
```

Open:

```
http://localhost:8080
```

---

## 🎯 Use Case

HeartScape is designed primarily for:

* Medical education
* Congenital heart disease visualization
* Data-driven anatomy exploration
* Hackathon analytics demonstrations

---

## 👥 Contributors

* Soham Agrawal
* Atharv Singh
* Shreya Chakraborty
* Vehnil Rangaraman

---

## 📜 License

This project is licensed under the Apache 2.0 License.

---

## 💡 Future Improvements

* Real-time medical imaging uploads (DICOM support)
* RAG-based medical dataset integration
* Multi-condition comparison overlays
* Cloud deployment with persistent storage

---

### Built for Hacklytics 2026 🚀