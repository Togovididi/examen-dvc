import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
PROC_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'

def main():
    X_train = pd.read_csv(os.path.join(PROC_DIR, 'X_train_scaled.csv'))
    y_train = np.ravel(pd.read_csv(os.path.join(PROC_DIR, 'y_train.csv')))
    best_params = joblib.load(os.path.join(MODELS_DIR, 'best_params.pkl'))
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODELS_DIR, 'model.pkl'))
    print('Modèle entraîné et sauvegardé (models/model.pkl) :', best_params)
if __name__ == '__main__':
    main()
