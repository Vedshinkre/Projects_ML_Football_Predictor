"""
Ensemble Tree Model Benchmarking & Validation.

This script evaluates tree-based ensemble methods (Random Forest and Gradient Boosting) 
across our feature sets. 

Sports data is notoriously noisy, and deep decision trees are highly prone to overfitting 
(memorizing the training data instead of learning general rules). Therefore, we strictly 
constrain tree depth (max_depth) to act as regularization, forcing the models to focus 
only on the most robust signals.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local project imports
from config import TRAIN_DATA_PATH, FEATURE_SETS
from feature_engineering import prepare_data

def run_trees_cv():
    print("Loading Data for Robust Tree Experiments (Cross-Validation)...")
    
    # ---------------------------------------------------------
    # 1. DATA INGESTION & PREPARATION
    # ---------------------------------------------------------
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df, _ = prepare_data(raw_df)
    
    # Note: While advanced libraries like XGBoost handle NaNs natively, 
    # standard Scikit-Learn implementations of RF and GB require complete data.
    full_df = full_df.dropna()
    
    print("\n============================================================")
    print("STARTING TREE CROSS-VALIDATION (5-FOLD TIME SERIES)")
    print("============================================================")
    
    results = []

    # ---------------------------------------------------------
    # 2. MODEL DEFINITIONS (With Regularization)
    # ---------------------------------------------------------
    models = {
        # Random Forest: Builds multiple independent trees and averages their votes.
        # max_depth=5 prevents the trees from learning overly specific, non-repeatable match events.
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        
        # Gradient Boosting: Builds trees sequentially, where each tree corrects the errors of the last.
        # A lower learning_rate (0.05) combined with shallow trees (max_depth=4) prevents rapid overfitting.
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    

    # ---------------------------------------------------------
    # 3. VALIDATION STRATEGY
    # ---------------------------------------------------------
    # Enforcing temporal integrity to prevent future data leakage
    tscv = TimeSeriesSplit(n_splits=5)

    # ---------------------------------------------------------
    # 4. FEATURE SET ITERATION & EVALUATION
    # ---------------------------------------------------------
    for level_name, features in FEATURE_SETS.items():
        print(f"\n--- Testing Feature Set: {level_name} ---")
        
        # Safety check for missing columns
        missing = [c for c in features if c not in full_df.columns]
        if missing:
            print(f"  Skipping {level_name}: Missing columns {missing}")
            continue
            
        X = full_df[features]
        y = full_df["Target"]
        
        for model_name, model in models.items():
            # Note: Tree models do not strictly *require* feature scaling, as they split 
            # on values regardless of scale. However, keeping StandardScaler in the pipeline 
            # ensures identical data processing across all our experiment scripts for fair comparison.
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
            
            # Execute cross-validation
            cv_acc = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
            cv_prec = cross_val_score(pipeline, X, y, cv=tscv, scoring='precision')
            
            avg_acc = cv_acc.mean()
            avg_prec = cv_prec.mean()
            
            print(f"   > {model_name}: {avg_acc:.1%} Acc (Avg) | {avg_prec:.1%} Prec (Avg)")
            
            # Record metrics
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
    print("ROBUST TREE LEADERBOARD (Sorted by Accuracy)")
    
    results_df = pd.DataFrame(results).sort_values(by="Avg Accuracy", ascending=False)
    print(results_df[['Feature Set', 'Model', 'Avg Accuracy', 'Avg Precision', 'Min Accuracy']])
    print("============================================================")
    
    # Ensure directory exists and save to the standardized results folder
    os.makedirs("data/results", exist_ok=True)
    results_df.to_csv("data/results/robust_tree_results.csv", index=False)
    print("Results saved successfully to data/results/robust_tree_results.csv")

if __name__ == "__main__":
    run_trees_cv()