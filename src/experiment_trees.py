"""
Advanced Tree Model Experimentation & Hyperparameter Tuning.

This script executes a systematic Grid Search to find the optimal hyperparameters 
for Random Forest and Gradient Boosting models across our progressively complex feature sets. 

Like previous scripts, it enforces a strict 80/20 chronological Train/Test split. 
The tuning focuses heavily on regularization parameters (max_depth, min_samples_split) 
to prevent the trees from overfitting the noisy sports data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

# Local project imports
from config import FEATURE_SETS, TRAIN_DATA_PATH
from feature_engineering import prepare_data

def run_tree_experiments():
    print("Loading Data for Tuned Tree Experiments...")
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # ---------------------------------------------------------
    # 1. DATA PREPARATION & CHRONOLOGICAL SPLIT
    # ---------------------------------------------------------
    full_df, _ = prepare_data(raw_df)
    
    # Standard Scikit-Learn tree implementations require complete data (no NaNs)
    full_df = full_df.dropna()
    
    # Split chronologically (80% Train, 20% Test) to prevent data leakage
    split_idx = int(len(full_df) * 0.80)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    y_train = train_df["Target"]
    y_test = test_df["Target"]
    
    print(f"Training on {len(train_df)} historical games. Testing on {len(test_df)} recent games.")

    # ---------------------------------------------------------
    # 2. MODEL & HYPERPARAMETER GRID DEFINITIONS
    # ---------------------------------------------------------
    models = [
        {
            "name": "Random Forest",
            "estimator": Pipeline([
                # While trees are scale-invariant, scaling is kept to ensure a 1-to-1 
                # comparison with our Logistic Regression experiments.
                ("scaler", StandardScaler()), 
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                # Test different ensemble sizes
                "clf__n_estimators": [100, 200, 300],
                
                # Test deeper trees to capture complex non-linear interactions 
                # (e.g., Elo Rating vs Bookmaker Odds gaps)
                "clf__max_depth": [5, 10, 15], 
                
                # min_samples_split acts as strong regularization, preventing the tree 
                # from splitting into highly specific, unrepeatable single-game events
                "clf__min_samples_split": [5, 10]
            }
        },
        {
            "name": "Gradient Boosting",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=42))
            ]),
            "params": {
                "clf__n_estimators": [100, 200],
                # Test slower learning rates to prevent the model from rapidly chasing noise
                "clf__learning_rate": [0.01, 0.05],
                # Keep trees shallow; GB relies on sequential correction, not deep individual trees
                "clf__max_depth": [3, 5]
            }
        }
    ]

    print(f"\n{'='*60}")
    print(f"STARTING TUNED TREE MODEL TOURNAMENT")
    print(f"{'='*60}")

    results = []

    # ---------------------------------------------------------
    # 3. EXPERIMENT LOOP (Iterating through Feature Levels)
    # ---------------------------------------------------------
    for dataset_name, cols in FEATURE_SETS.items():
        print(f"\n--- Testing Feature Set: {dataset_name} ---")
        
        # Safety check: Ensure configuration matches data
        missing = [c for c in cols if c not in full_df.columns]
        if missing:
            print(f"  Skipping {dataset_name}: Missing columns {missing}")
            continue

        X_train = train_df[cols]
        X_test = test_df[cols]

        for m in models:
            print(f"   > Tuning {m['name']}...")
            
            # TimeSeriesSplit inside GridSearch ensures we never tune parameters using future data
            cv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(m['estimator'], m['params'], cv=cv, scoring='accuracy', n_jobs=-1)
            
            try:
                # 1. Find optimal hyperparameters using the Training set
                grid.fit(X_train, y_train)
                
                # 2. Extract best model configuration
                best_model = grid.best_estimator_
                
                # 3. Generate predictions on the unseen Test set
                preds = best_model.predict(X_test)
                
                # 4. Evaluate and store metrics
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
        df_results.to_csv("data/results/tuned_tree_results.csv", index=False)
        
        print("\n" + "="*60)
        print("TUNED TREE LEADERBOARD")
        print(df_results[["Feature Set", "Model", "Test Accuracy", "Precision"]])
        print("="*60)

if __name__ == "__main__":
    run_tree_experiments()