import sys
from pathlib import Path
import pandas as pd
import joblib

# Dynamically locate project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_prep import clean_data, engineer_features


def main():
    model_path = PROJECT_ROOT / "model.pkl"

    if not model_path.exists():
        print("ERROR: model.pkl not found. Run 'python src/train.py' first.")
        return

    print("Loading trained model...")
    model = joblib.load(model_path)

    # 1. Predict for November Validation (12,000 loads)
    val_csv_path = PROJECT_ROOT / "data" / "validation.csv"
    template_csv_path = PROJECT_ROOT / "data" / "validation-predictions-template.csv"
    out_val_path = PROJECT_ROOT / "validation_predictions.csv"

    print("Generating validation_predictions.csv...")
    val_df = clean_data(val_csv_path)
    val_df, features = engineer_features(val_df)

    val_template = pd.read_csv(template_csv_path)
    val_template["predicted_rate"] = model.predict(val_df[features])
    val_template.to_csv(out_val_path, index=False)
    print(f"Saved: {out_val_path}")

    # 2. Predict for December Chart Inputs (31 loads)
    dec_csv_path = PROJECT_ROOT / "data" / "december-chart-inputs.csv"

    print("Updating december-chart-inputs.csv...")
    dec_df = clean_data(dec_csv_path)
    dec_df, features = engineer_features(dec_df)

    dec_df["predicted_rate"] = model.predict(dec_df[features])

    # Convert date back to string format for score.py
    dec_df["date"] = pd.to_datetime(dec_df["date"]).dt.strftime("%Y-%m-%d")

    # Strictly keep original 7 columns required by score.py
    required_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_df[required_cols].to_csv(dec_csv_path, index=False)
    print(f"Updated: {dec_csv_path}")

    print("\nAll predictions complete!")


if __name__ == "__main__":
    main()