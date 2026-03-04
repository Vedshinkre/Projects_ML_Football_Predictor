"""
Basic Model Experimentation & Hyperparameter Tuning.

This script evaluates traditional machine learning algorithms across different feature sets.
Unlike basic cross-validation, this script uses GridSearchCV to find the optimal 
hyperparameters for each model. 

It enforces a strict chronological 80/20 Train/Test split. The grid search optimizes 
the model using only the past 80% of data, and the final evaluation is performed 
on the most recent 20% of matches to simulate real-world betting conditions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

# Local project imports
from config import FEATURE_SETS, TRAIN_DATA_PATH
from feature_engineering import prepare_data

def run_basic_experiments():
    print("Loading Data for Basic Experiments with Hyperparameter Tuning...")
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # ---------------------------------------------------------
    # 1. DATA PREPARATION & CHRONOLOGICAL SPLIT
    # ---------------------------------------------------------
    # Process all features (Elo, H2H, xGD, Odds, etc.)
    full_df, _ = prepare_data(raw_df)
    
    # Drop NaNs required for Logistic Regression and SVMs
    full_df = full_df.dropna()
    
    # Strict chronological split (80% Train, 20% Test). 
    # We do NOT use train_test_split(random_state) because randomly shuffling 
    # time-series data causes future data leakage.
    split_idx = int(len(full_df) * 0.80)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    y_train = train_df["Target"]
    y_test = test_df["Target"]
    
    print(f"Training on {len(train_df)} historical games. Testing on {len(test_df)} recent games.")

    # ---------------------------------------------------------
    # 2. MODEL & HYPERPARAMETER GRID DEFINITIONS
    # ---------------------------------------------------------
    # We define pipelines to scale data correctly inside the cross-validation loop,
    # paired with a dictionary of parameters to test.
    models = [
        {
            "name": "Logistic Regression",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                # High max_iter required for convergence with complex features like Elo/Worth
                ("clf", LogisticRegression(max_iter=5000, random_state=42)) 
            ]),
            "params": {
                # Test different regularization strengths
                "clf__C": [0.001, 0.01, 0.1, 1],
                # 'liblinear' is highly robust for datasets under 100k rows
                "clf__solver": ["liblinear"] 
            }
        },
        {
            "name": "KNN (Neighbors)",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "params": {
                # Football is high-variance; higher neighbor counts prevent fitting to noise
                "clf__n_neighbors": [15, 30, 50], 
                "clf__weights": ["uniform", "distance"]
            }
        },
        {
            "name": "SVM (Support Vector)",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=42, probability=True))
            ]),
            "params": {
                "clf__C": [0.1, 1],
                # RBF kernel helps capture non-linear relationships (e.g., Elo vs Worth gaps)
                "clf__kernel": ["rbf"] 
            }
        }
    ]

    print(f"\n{'='*60}")
    print(f"STARTING BASIC MODEL TOURNAMENT")
    print(f"{'='*60}")

    results = []
    
    # ---------------------------------------------------------
    # 3. EXPERIMENT LOOP (Iterating through Feature Levels)
    # ---------------------------------------------------------
    for dataset_name, cols in FEATURE_SETS.items():
        print(f"\n--- Testing Feature Set: {dataset_name} ---")
        
        # Safety Check: Ensure the requested configuration matches the generated dataframe
        missing = [c for c in cols if c not in full_df.columns]
        if missing:
            print(f"  Skipping {dataset_name}: Missing columns {missing}")
            continue
            
        X_train = train_df[cols]
        X_test = test_df[cols]

        for m in models:
            print(f"   > Tuning {m['name']}...")
            
            # TimeSeriesSplit inside the GridSearch prevents validation leakage
            cv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(m['estimator'], m['params'], cv=cv, scoring='accuracy', n_jobs=-1)
            
            try:
                # 1. Find the best hyperparameters using ONLY the training data
                grid.fit(X_train, y_train)
                
                # 2. Extract the best model from the grid search
                best_model = grid.best_estimator_
                
                # 3. Predict on the completely unseen Test dataset
                preds = best_model.predict(X_test)
                
                # 4. Evaluate performance
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                
                results.append({
                    "Feature Set": dataset_name,
                    "Model": m['name'],
                    "Test Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Best Params": str(grid.best_params_)
                })
                print(f"     -> Val Acc: {grid.best_score_:.1%} | Test Acc: {acc:.1%} | Test Prec: {prec:.1%}")
            except Exception as e:
                print(f"     Failed: {e}")

    # ---------------------------------------------------------
    # 4. FINAL LEADERBOARD & EXPORT
    # ---------------------------------------------------------
    if results:
        df_results = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
        
        os.makedirs("data/results", exist_ok=True)
        df_results.to_csv("data/results/tuned_basic_results.csv", index=False)
        
        print("\n" + "="*60)
        print("TUNED BASIC LEADERBOARD")
        print(df_results[["Feature Set", "Model", "Test Accuracy", "Precision"]])
        print("="*60)

if __name__ == "__main__":
    run_basic_experiments()