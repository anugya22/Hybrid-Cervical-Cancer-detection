# data_load.py
# I load the dataset CSV and prepare train/validation splits + a scaled tabular feature array.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_prepare(csv_path, image_col='image_path', label_col='label', tabular_cols=None, test_size=0.2, random_state=42):
    """
    Returns:
      df: original dataframe
      all_tab: numpy array of tabular features aligned to df.index
      train_df: dataframe for training (contains original index in column 'orig_index')
      val_df: dataframe for validation (contains original index in column 'orig_index')
      tabular_cols: list of used tabular column names
    Notes:
      - I keep original df.index as a column 'orig_index' so we can map tabular rows reliably.
    """
    df = pd.read_csv(csv_path)
    # create an explicit original index column to avoid confusion after resets
    df = df.reset_index().rename(columns={'index': 'orig_index'})

    if tabular_cols is None:
        tabular_cols = [c for c in df.columns if c not in ['orig_index', image_col, label_col]]

    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=random_state)

    # global tabular array aligned by 'orig_index' (orig_index -> row in array)
    all_tab = df[tabular_cols].fillna(0).values.astype('float32')

    # Fit scaler on train only, then transform global array
    scaler = StandardScaler()
    scaler.fit(train_df[tabular_cols].fillna(0).values)
    all_tab = scaler.transform(df[tabular_cols].fillna(0).values).astype('float32')

    return df, all_tab, train_df, val_df, tabular_cols
