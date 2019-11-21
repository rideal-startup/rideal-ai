import click

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def grid():
    return {
        'learning_rate': np.linspace(start=0.1, stop=0.5, num=5).tolist(),
        'max_depth': np.linspace(10, 110, num=10).astype('int').tolist(),
        'colsample_bytree': np.linspace(0.2, 0.6, num=5).tolist(),
        'n_estimators': np.linspace(5, 20, num=10).astype('int')
    }


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
    y_train = train_df['target'].values.reshape(-1, 1)
    y_train = LabelEncoder().fit_transform(y_train)

    X_test = test_df.drop('target', axis='columns').values
    y_test = test_df['target'].values.reshape(-1, 1)
    y_test = LabelEncoder().fit_transform(y_test)

    train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

    rnd_search = RandomizedSearchCV(estimator=xgb.XGBClassifier(),
                                    param_distributions=grid(),
                                    n_iter=100,
                                    cv=5, random_state=seed)
    rnd_search.fit(X_train, y_train)
    model = rnd_search.best_estimator_
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
