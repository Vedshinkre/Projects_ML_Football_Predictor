"""
Hyperparameter Tuning Module.

This script utilizes RandomizedSearchCV to efficiently discover the optimal 
hyperparameters for the Level 6 (Ultimate) Random Forest model. 

It strictly enforces temporal integrity by using a TimeSeriesSplit. This ensures 
that during the cross-validation process, the model is only ever validated on 
future data relative to its training folds, completely preventing data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local project imports
from config import TRAIN_DATA_PATH, FEATURE_SETS
from feature_engineering import prepare_data

def run_hyperparameter_tuning():
    print("STARTING RANDOM FOREST HYPERPARAMETER TUNING (LEVEL 6)")
    
    # ---------------------------------------------------------
    # 1. Load & Prepare Data
    # ---------------------------------------------------------
    print(" -> Loading and preparing historical data...")
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df, _ = prepare_data(raw_df)
    
    # ---------------------------------------------------------
    # 2. Select Features
    # ---------------------------------------------------------
    features = FEATURE_SETS["Level 6 (Ultimate)"]
    print(f" -> Tuning utilizing {len(features)} engineered features.")
    
    # Standard Scikit-Learn tree implementations require complete data
    full_df = full_df.dropna(subset=features)
    
    X = full_df[features]
    y = full_df["Target"]
    
    # ---------------------------------------------------------
    # 3. Define the Hyperparameter Search Space
    # ---------------------------------------------------------
    # Instead of an exhaustive Grid Search (which is computationally expensive),
    # RandomizedSearchCV will sample 50 combinations from this distribution.
    param_grid = {
        'clf__n_estimators': [100, 200, 300, 500],         # Number of trees in the forest
        'clf__max_depth': [5, 10, 15, None],               # Maximum depth of the tree
        'clf__min_samples_split': [2, 5, 10, 20],          # Minimum samples required to split an internal node
        'clf__min_samples_leaf': [1, 2, 4, 8],             # Minimum samples required to be at a leaf node
        'clf__max_features': ['sqrt', 'log2', None]        # Number of features to consider when looking for the best split
    }
    
    # ---------------------------------------------------------
    # 4. Construct the Pipeline
    # ---------------------------------------------------------
    # The scaler is included in the pipeline to prevent data leakage between folds
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # ---------------------------------------------------------
    # 5. Execute Search with Temporal Validation
    # ---------------------------------------------------------
    print(" -> Initializing TimeSeriesSplit and RandomizedSearchCV...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    search = RandomizedSearchCV(
        pipeline, 
        param_grid, 
        n_iter=50,       # Randomly sample 50 configurations to save compute time
        cv=tscv,         # Enforce chronological validation
        scoring='accuracy', 
        n_jobs=-1,       # Utilize all available CPU cores
        verbose=1,
        random_state=42
    )
    
    print(" -> Executing search grid (this may take several minutes)...")
    search.fit(X, y)
    
    # ---------------------------------------------------------
    # 6. Output Results
    # ---------------------------------------------------------
    print("\nTUNING COMPLETE.")
    print(f" -> Best Cross-Validated Accuracy: {search.best_score_:.1%}")
    print(" -> Optimal Parameters (Transfer these to train_model.py):")
    print(search.best_params_)

if __name__ == "__main__":
    run_hyperparameter_tuning()