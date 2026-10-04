import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
PROC_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'

def main():
    X_train = pd.read_csv(os.path.join(PROC_DIR, 'X_train_scaled.csv'))
    y_train = np.ravel(pd.read_csv(os.path.join(PROC_DIR, 'y_train.csv')))
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(grid.best_params_, os.path.join(MODELS_DIR, 'best_params.pkl'))
    print('Meilleurs paramètres :', grid.best_params_)
    print('Meilleur score CV (r2) :', grid.best_score_)
if __name__ == '__main__':
    main()
