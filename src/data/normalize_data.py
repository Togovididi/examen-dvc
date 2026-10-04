import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
PROC_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'

def main():
    X_train = pd.read_csv(os.path.join(PROC_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROC_DIR, 'X_test.csv'))
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_train_scaled.to_csv(os.path.join(PROC_DIR, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(PROC_DIR, 'X_test_scaled.csv'), index=False)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print('Normalisation OK (X_train_scaled, X_test_scaled)')
if __name__ == '__main__':
    main()
