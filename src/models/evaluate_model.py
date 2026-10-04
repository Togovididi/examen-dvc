import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
PROC_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'
METRICS_DIR = 'metrics'
DATA_DIR = 'data'

def main():
    X_test = pd.read_csv(os.path.join(PROC_DIR, 'X_test_scaled.csv'))
    y_test = np.ravel(pd.read_csv(os.path.join(PROC_DIR, 'y_test.csv')))
    model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
    y_pred = model.predict(X_test)
    predictions = X_test.copy()
    predictions['y_true'] = y_test
    predictions['y_pred'] = y_pred
    os.makedirs(DATA_DIR, exist_ok=True)
    predictions.to_csv(os.path.join(DATA_DIR, 'predictions.csv'), index=False)
    mse = float(mean_squared_error(y_test, y_pred))
    scores = {'mse': mse, 'rmse': float(mse ** 0.5), 'mae': float(mean_absolute_error(y_test, y_pred)), 'r2': float(r2_score(y_test, y_pred))}
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    print('Scores :', scores)
if __name__ == '__main__':
    main()
