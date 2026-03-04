# Premier League Match Predictor: A True Fundamental Model

## 🔄 Project Evolution: From Beta to V1.0

This repository is the successor to **Projects_ML_Football_Predictor_Beta**.  
To make this system highly reproducible and aligned with industry MLOps standards, the following architectural upgrades were implemented:

- **Jupyter → Python Architecture**  
  Transitioned from static Jupyter Notebooks to a modular Python architecture (`src/` scripts) for improved reproducibility, version control clarity, and easier deployment.

- **Optimized Data Ingestion**  
  Replaced brittle web-scraping scripts with highly reliable, publicly available historical CSV datasets.

- **Expanded Feature Space**  
  Engineered advanced football metrics, including:
  - Dynamic Elo ratings  
  - Expected Goal Differential (xGD)  
  - Log-scaled financial gaps between squads  

- **Rigorous Experimentation Framework**  
  Replaced basic modeling workflows with a structured **"Feature Staircase" ablation study**, utilizing chronological cross-validation to eliminate data leakage entirely.

- **Interactive Front-End Deployment**  
  Built and deployed a fully functional Streamlit web application for real-time, in-memory inference.

---
## 📌 Project Overview

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict English Premier League match outcomes.

This project achieves a **66.5% cross-validated accuracy** using a strictly independent, fundamental approach. The model relies purely on mathematical team strength, tactical form, historical matchups, and EA Sports FIFA squad financial valuations.

---

## ⭐ Key Features

- **Dynamic Memory State:**  
  Implements a custom chronological engine to calculate running Elo ratings and Head-to-Head win percentages without future data leakage.

- **Entity Resolution:**  
  Merges live match data from Football-Data.co.uk with EA Sports FIFA squad valuations to quantify the financial gap between clubs.

- **Hyperparameter Optimization:**  
  Utilizes `RandomizedSearchCV` paired with `TimeSeriesSplit` to tune the Random Forest architecture while respecting the temporal nature of sports data.

- **Production Web App:**  
  Includes a fully functional Streamlit interface that caches serialized model artifacts (`.pkl` files) for millisecond inference on upcoming fixtures.

---

## 📁 Project Structure

```text
football-predictor/
├── data/
│   ├── raw/                  # Raw CSV files from football-data.co.uk & FIFA
│   └── results/              # Experiment leaderboards and validation metrics
├── models/                   # Serialized artifacts (final_model.pkl, elo_dict.pkl)
├── src/
│   ├── app.py                # Streamlit web application interface
│   ├── config.py             # Centralized feature set configurations
│   ├── feature_engineering.py# Pipeline for Elo, xGD, and rolling stats
│   ├── experiment_*.py       # Cross-validation and tuning tournament scripts
│   ├── train_model.py        # Production model training & memory capture script
│   └── tune_model.py         # Hyperparameter tuning
├── README.md
├── REPORT.md                 # Detailed technical breakdown of experiments
└── requirements.txt
```

---

## 🚀 Quick Start Guide

### 1️⃣ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/football-predictor.git
cd football-predictor
pip install -r requirements.txt
```

---

### 2️⃣ Tuning & Training the Model

To replicate the AI's "Brain", run the hyperparameter tuner followed by the training script.

This will populate the `models/` directory with the required serialized dictionaries.

```bash
python src/tune_model.py
python src/train_model.py
```

---

### 3️⃣ Launching the Web Application

Start the Streamlit server to interact with the model and generate live predictions:

```bash
streamlit run src/app.py
```

---

## 🧠 Technical Deep Dive

For a comprehensive breakdown of:

- The feature engineering pipeline  
- The chronological cross-validation strategy  
- The multi-level experiment tournament  

Please refer to the technical report:

📄 **[REPORT.md](REPORT.md)**