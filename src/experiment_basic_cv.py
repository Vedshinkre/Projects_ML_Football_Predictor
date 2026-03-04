"""
Baseline Model Benchmarking & Validation.

This script evaluates traditional machine learning algorithms (Logistic Regression, 
KNN, SVM) across our progressively complex feature sets. 

It utilizes a Time Series Split for cross-validation to ensure temporal integrity 
(preventing future data from leaking into past predictions) and employs a Pipeline 
to prevent data leakage during feature scaling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local project imports
from config import TRAIN_DATA_PATH, FEATURE_SETS
from feature_engineering import prepare_data

def run_basic_cv():
    print("Loading Data for Robust Basic Experiments (Cross-Validation)...")
    
    # ---------------------------------------------------------
    # 1. DATA INGESTION & PREPARATION
    # ---------------------------------------------------------
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df, _ = prepare_data(raw_df)
    
    # Traditional Scikit-Learn models (unlike XGBoost) cannot natively handle NaN values.
    # We must drop rows with missing values to ensure stability across all algorithms.
    full_df = full_df.dropna()
    
    print("\n============================================================")
    print("STARTING BASIC CROSS-VALIDATION (5-FOLD TIME SERIES)")
    print("============================================================")
    
    results = []

    # ---------------------------------------------------------
    # 2. MODEL DEFINITIONS
    # ---------------------------------------------------------
    # We establish baselines using linear and distance-based classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (Neighbors)": KNeighborsClassifier(n_neighbors=15),
        "SVM (Support Vector)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    }
    
    # ---------------------------------------------------------
    # 3. VALIDATION STRATEGY
    # ---------------------------------------------------------
    # TimeSeriesSplit ensures that the model is always trained on historical data 
    # and evaluated on future data, mimicking real-world betting scenarios.
    tscv = TimeSeriesSplit(n_splits=5)

    # ---------------------------------------------------------
    # 4. FEATURE SET ITERATION & EVALUATION
    # ---------------------------------------------------------
    for level_name, features in FEATURE_SETS.items():
        print(f"\n--- Testing Feature Set: {level_name} ---")
        
        # Safety check: Ensure all configured features exist in the current dataframe
        missing = [c for c in features if c not in full_df.columns]
        if missing:
            print(f"  Skipping {level_name}: Missing columns {missing}")
            continue
            
        X = full_df[features]
        y = full_df["Target"]
        
        for model_name, model in models.items():
            # Standardize features (mean=0, variance=1) inside a Pipeline.
            # Doing this inside the Pipeline ensures the scaler is only fit on the 
            # training fold, preventing data leakage from the validation fold.
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
            
            # Execute cross-validation for multiple metrics
            cv_acc = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
            cv_prec = cross_val_score(pipeline, X, y, cv=tscv, scoring='precision')
            
            avg_acc = cv_acc.mean()
            avg_prec = cv_prec.mean()
            
            print(f"   > {model_name}: {avg_acc:.1%} Accuracy (Avg)")
            
            # Store results for leaderboard comparison
            results.append({
                "Feature Set": level_name,
                "Model": model_name,
                "Avg Accuracy": avg_acc,
                "Avg Precision": avg_prec,
                "Min Accuracy": cv_acc.min(),
                "Max Accuracy": cv_acc.max()
            })

    # ---------------------------------------------------------
    # 5. REPORTING & EXPORT
    # ---------------------------------------------------------
    print("\n============================================================")
    print("ROBUST BASIC LEADERBOARD (Sorted by Accuracy)")
    
    results_df = pd.DataFrame(results).sort_values(by="Avg Accuracy", ascending=False)
    print(results_df[['Feature Set', 'Model', 'Avg Accuracy', 'Avg Precision', 'Min Accuracy']])
    print("============================================================")
    
    # Ensure the results directory exists before saving
    os.makedirs("data/results", exist_ok=True)
    results_df.to_csv("data/results/robust_basic_results.csv", index=False)
    print("Results saved successfully to data/results/robust_basic_results.csv")

if __name__ == "__main__":
    run_basic_cv()