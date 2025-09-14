# data_load.py
# I load the dataset CSV and prepare train/validation splits + a scaled tabular feature array.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_prepare(csv_path, image_col='image_path', label_col='label', tabular_cols=None, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    if tabular_cols is None:
        tabular_cols = [c for c in df.columns if c not in [image_col, label_col]]
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=random_state)
    all_tab = df[tabular_cols].fillna(0).values.astype('float32')
    scaler = StandardScaler()
    scaler.fit(train_df[tabular_cols].fillna(0).values)
    all_tab = scaler.transform(df[tabular_cols].fillna(0).values).astype('float32')
    # Return original df and global tabular array (aligned by df.index), plus splits
    return df, all_tab, train_df, val_df, tabular_cols
