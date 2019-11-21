import click

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def hyperparameter_grid():
    n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=20)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=20)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    return random_grid


@click.command()
@click.option('--train',
              help='Path to the train dataset')
@click.option('--test',
              help='Path to the test dataset')
@click.option('--seed',
              help='Random seed',
              default=42)
def main(train, test, seed):
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    X_train = train_df.drop('target', axis='columns').values
    y_train = train_df['target'].values.reshape(-1)

    X_test = test_df.drop('target', axis='columns').values
    y_test = test_df['target'].values.reshape(-1)

    grid = hyperparameter_grid()
    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(), 
                                   param_distributions=grid, 
                                   n_iter=100, 
                                   cv=5, random_state=seed)
    rf_random.fit(X_train, y_train)
    best_model = rf_random.best_estimator_
    print(classification_report(y_test, rf_random.predict(X_test)))


if __name__ == "__main__":
    main()