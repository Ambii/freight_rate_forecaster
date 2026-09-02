import pandas as pd

def clean_data(file_path="data/train-test.csv"):
    """Loads raw CSV data and handles missing values/formatting."""
    df = pd.read_csv(file_path)
    
    # Impute missing weights with column median
    if 'weight' in df.columns:
        df['weight'] = df['weight'].fillna(df['weight'].median())
        
    # Convert date string to datetime object
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    return df


def engineer_features(df):
    """Extracts date features and converts categorical types."""
    # Temporal features present across all test/val splits
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.weekday
    
    # Convert text columns to categorical for LightGBM
    categorical_cols = ['pickup', 'delivery', 'equipment']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Shared feature set available in all evaluation datasets
    features = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'month', 'day_of_week']
    
    return df, features