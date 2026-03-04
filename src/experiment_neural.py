import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import TRAIN_DATA_PATH, FEATURE_SETS
from feature_engineering import prepare_data

def run_neural_experiment():
    print("Loading Data for NEURAL NETWORK Experiment...")
    
    # 1. Load Data
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df, _ = prepare_data(raw_df)
    
    # Use Level 7 (Context) or Level 6 (Ultimate) - whichever you added last
    # Let's default to Level 6 if you haven't added Level 7 yet
    target_level = "Level 7 (Context)" if "Level 7 (Context)" in FEATURE_SETS else "Level 6 (Ultimate)"
    features = FEATURE_SETS[target_level]
    
    print(f"   -> Using Feature Set: {target_level}")
    
    # Drop rows with missing values
    full_df = full_df.dropna(subset=features)
    X = full_df[features]
    y = full_df["Target"]
    
    print(f"\n============================================================")
    print(f"NEURAL NET vs GRADIENT BOOSTING (5-FOLD CV)")
    print(f"============================================================")
    
    # 2. Define Models
    models = {
        # The Current Champion
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        ),
        
        # The Challenger: Neural Network (MLP)
        # hidden_layer_sizes=(64, 32): Two layers of "neurons"
        # max_iter=1000: Give it time to learn
        # alpha=0.001: Regularization (prevents overfitting)
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            solver='adam', 
            alpha=0.001, 
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=1000, 
            random_state=42,
            early_stopping=True # Stop if it stops improving (saves time)
        )
    }
    
    # 3. Validation
    tscv = TimeSeriesSplit(n_splits=5)

    for model_name, model in models.items():
        # Neural Nets MUST have scaled data (StandardScaler is crucial here)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        
        # Run CV
        cv_acc = cross_val_score(pipeline, X, y, cv=tscv, scoring='accuracy')
        cv_prec = cross_val_score(pipeline, X, y, cv=tscv, scoring='precision')
        
        print(f"\n   > {model_name}:")
        print(f"     -> Accuracy:  {cv_acc.mean():.1%} (Range: {cv_acc.min():.1%} - {cv_acc.max():.1%})")
        print(f"     -> Precision: {cv_prec.mean():.1%}")

if __name__ == "__main__":
    run_neural_experiment()