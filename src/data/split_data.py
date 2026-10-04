import os
import pandas as pd
from sklearn.model_selection import train_test_split
RAW_PATH = os.path.join('data', 'raw', 'raw.csv')
OUT_DIR = os.path.join('data', 'processed')
TARGET = 'silica_concentrate'

def main():
    df = pd.read_csv(RAW_PATH)
    for col in list(df.columns):
        if col.lower() in ('date', 'unnamed: 0', 'index'):
            df = df.drop(columns=[col])
    if TARGET in df.columns:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUT_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, 'y_test.csv'), index=False)
    print(f'Split OK : X_train={X_train.shape}, X_test={X_test.shape}')
if __name__ == '__main__':
    main()
