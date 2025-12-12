# Firearm Mortality & Gun Law Strength Dashboard

An interactive **Plotly Dash** dashboard + analysis notebooks exploring how **U.S. state firearm policy changes** (converted into a cumulative *law strength score*) relate to **firearm death rates** over time.

> Dashboard deployed here: https://cda296d2-afd5-4634-8d4c-1ae9fc74cee5.plotly.app/ 

---

## Repository structure

```text
.
├─ dashboard/                 # Dash app (UI + callbacks + local assets)
├─ Data/
│  ├─ raw/                    # Original source files (2 datasets)
│  └─ processed/              # Cleaned/merged dataset
├─ notebooks/
│  ├─ eda.ipynb               # Exploratory data analysis (pre-modeling)
│  └─ full_analysis.ipynb     # Modeling + interpretation
├─ scripts/
│  └─ clean_data.R            # Data cleaning + law scoring pipeline (R)
└─ Documentation.ipynb        # Data engineering / scoring documentation
```

---

## Dashboard (Plotly Dash)

**Entry point:** `dashboard/app.py` 

### What the dashboard includes
Tabs currently implemented: **Introduction**, **Data Table** (scroll + filter), **State Maps**, **Predicting Gun Violence**, **Effectiveness Metrics**, **State Clusters**, **Data Sources**.

### Run locally

From the repo root:

```bash
python dashboard/app.py
```

### Required dashboard files

The app reads these markdown files at runtime (expected to be in the same working directory as `app.py`):
- `intro.md`
- `datasources.md`
- `research_q1.md`
- `research_q1_bottom.md`

It also loads precomputed model output CSVs for the “Predicting Gun Violence” tab (expected alongside `app.py`): 
- `basic_output_data1.csv`, `ridge_output_data1.csv`, `pca_output_data1.csv`
- `basic_output_data2.csv`, `ridge_output_data2.csv`, `pca_output_data2.csv`

And it displays local images via Dash’s `assets/` mechanism:
- `kmeans_elbow.png`
- `ward_dendrogram.png`
- `pca_clusters.png`

---

## Project questions (from the full analysis)

The modeling work in `notebooks/full_analysis.ipynb` focuses on:

1. Can we predict firearm death rates based on gun law characteristics?
2. Which specific law types are most strongly associated with death rates?
3. Can we identify distinct groups of states based on gun law profiles?
4. How have these relationships evolved over time?

Methods used include **Linear Regression**, **Ridge** / **Lasso**, **KNN**, **K-Means**, **Hierarchical Clustering**, **PCA**, and an **MLP (Neural Network)** classifier.

---

## Data

### Raw (Data/raw/)
Two source datasets are used (see `datasources.md`):
- **CDC mortality** (`data-table.csv`) — firearm death rates / deaths by state-year
- **RAND State Firearm Law Database** (`TL-A243-2-v3 State Firearm Law Database 5.0.xlsx`) — law changes by state with `effect`, `type_of_change`, and `effective_date_year`

### Processed (Data/processed/)
- `firearm_data_cleaned.csv` — merged, cleaned, analysis-ready dataset loaded by the dashboard fileciteturn1file0  

---

## Law strength scoring (high level)

Each law change is mapped to **+1 / -1** and then **cumulatively summed** by state-year to represent the policy environment.

Scoring rules: fileciteturn1file1
- Restrictive **Implement/Modify** → **+1**
- Permissive **Implement/Modify** → **-1**
- Repeal of a restrictive law → **-1**
- Repeal of a permissive law → **+1**

The pipeline also creates per-class cumulative features (`strength_<law_class>`) and year-over-year change variables (`rate_change`, `law_strength_change`). 

Full details and rationale are documented in **`Documentation.ipynb`** and `scripts/clean_data.R`.

---

## Reproduce the dataset (R)

1) Place the raw source files into `Data/raw/`.  
2) Run:

```bash
Rscript scripts/clean_data.R
```

Expected output: a cleaned/merged CSV written to `Data/processed/`.

---

## Notebooks (Python)

- `notebooks/eda.ipynb` — EDA used to understand trends/distributions before modeling
- `notebooks/full_analysis.ipynb` — modeling workflow + interpretation

Run with:

```bash
jupyter lab
# or
jupyter notebook
```

---

## Contributors

- Chloe Barnes  
- Ryan Dallas  
- Terry Luedtke  
- Shiraz Robinson  
- Tiandre Threat fileciteturn1file2

---

## Notes

- This repository is intended for educational/research use.
- Results shown are **associations**, not causal estimates.
- Please cite the original data sources when using results or figures from this project.
