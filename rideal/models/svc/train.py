import click

import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import classification_report


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

    X_test = test_df.drop('target', axis='columns').values
    y_test = test_df['target'].values.reshape(-1, 1)
    
    model = SVC(random_state=seed)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()