import pandas as pd

def load_data(path):
    """Loads data from a given location and removes the 'datetime' column if it exists."""
    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])
    return df