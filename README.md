# Wood Surface Defect Detection
## Semester Project — Research Paper Implementation

**Based on:**
> Goycochea Casas, G.; Ismail, Z.H.; Leite, H.G.
> *Computer Vision, Machine Learning, and Deep Learning for Wood and Timber Products:
> A Scopus-Based Bibliometric and Systematic Mapping Review (1983–2026, Early Access).*
> Forests 2026, 17, 112. https://doi.org/10.3390/f17010112

---

## What This Project Does

This project implements the methodology and demonstrates the core findings of the above research paper by:

1. **Bibliometric Analysis (Paper Section 2.4 & 3)**: Re-implements the rule-based NLP classification from the paper to automatically classify 1,019 bibliographic records into application and methodological categories, generating exact replica charts.
2. **Classical CV vs Deep Learning Demonstration (Paper Section 5.1 & 5.3)**: Demonstrates the paper's key finding that *Wood Surface Quality Control* is the dominant application by providing a side-by-side comparison of Classical Computer Vision (Canny/Otsu) and Deep Learning (YOLOv8) on a sample dataset of wood defect images.

---

## Required Files

Before running the code, ensure the following datasets are in the project folder:

1. `data_clean.xlsx` — The 1019-paper bibliometric dataset (placed directly in the project root folder).
2. `dataset/` — A folder containing `.jpg` or `.png` wood surface defect images.
3. `models/` (Optional) — If you have trained a YOLOv8 model, place the `best.pt` file inside the `models/` directory for the script to load it for the Deep Learning demonstration.

---

## Setup Instructions

### Step 1 — Install Python dependencies
```bash
pip install pandas matplotlib seaborn opencv-python ultralytics openpyxl
```

### Step 2 — Run the Pipeline
Simply run `main.py` from your terminal. The script will automatically run both the bibliometric analysis and the computer vision demonstration.

```bash
python main.py
```

### Note on GPU Utilization
If you have a CUDA-compatible NVIDIA GPU, the `vision_pipeline.py` script is explicitly configured to use `device=0` during YOLOv8 inference, ensuring ultra-fast processing speeds.

---

## Project Structure

```
wood_defect_project/
│
├── main.py                      ← Run everything at once
├── config.py                    ← Configuration paths and settings
├── bibliometrics.py             ← Generates the NLP charts (Paper Section 2.4 & 3)
├── vision_pipeline.py           ← Runs CV vs DL comparison (Paper Section 5)
│
├── dataset/                     ← Put your raw wood images here
├── data_clean.xlsx              ← The OSF bibliometric data
│
├── models/
│   └── best.pt                  ← Optional: Trained YOLOv8 weights
│
└── outputs/
    ├── figures/                 ← Generated Bibliometric Charts (.png)
    └── results/                 ← Generated CV vs YOLO Comparison Images (.png)
```

---

## Paper-to-Code Mapping

| Paper Section | Our Code | What It Does |
|---------------|----------|--------------|
| Sec 2.4 (Rule-based classification) | `bibliometrics.py` | Assigns category labels using the paper's exact NLP heuristics. |
| Sec 3 (Bibliometric results) | `bibliometrics.py` | Reproduces paper Figs 2, 9, 10. |
| Sec 4.1 & 5.1 (Wood Surface QC / Classical CV) | `vision_pipeline.py` | Demonstrates Canny + Otsu + morphology for defect detection. |
| Sec 5.3 (Deep Learning) | `vision_pipeline.py` | Demonstrates YOLOv8 inference using your GPU. |
