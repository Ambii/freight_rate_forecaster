import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib

# Dynamically locate project root (one directory level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_prep import clean_data, engineer_features

# Paths locked to project root
DATA_PATH = PROJECT_ROOT / "data" / "train-test.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"

def main():
    print("Loading and cleaning training data...")
    df = clean_data(DATA_PATH)
    df, features = engineer_features(df)

    # Time-based split: Train (Jan-Sep), Validate (Oct)
    train_data = df[df['date'] < '2025-10-01']
    val_data = df[df['date'] >= '2025-10-01']

    X_train = train_data[features]
    y_train = train_data['posted_rate']

    X_val = val_data[features]
    y_val = val_data['posted_rate']

    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    mae = (y_val - predictions).abs().mean()
    print(f"Validation MAE: ${mae:.2f}")

    # Explicitly save model.pkl in project root
    joblib.dump(model, MODEL_PATH)
    print(f"SUCCESS: Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()