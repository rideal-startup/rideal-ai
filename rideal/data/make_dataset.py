from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.group()
def main():
    pass


@main.command()
@click.option('--src-file',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--dst',
              type=click.Path(file_okay=False))
@click.option('--seed', type=int, default=42)
def create_sets(src_file, dst, seed):
    df = pd.read_csv(src_file)
    train_df, test_df = train_test_split(df, 
                                         test_size=.3, 
                                         random_state=42)

    dst_path = Path(dst)
    dst_path.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(str(dst_path/'train.csv'), index=False)
    test_df.to_csv(str(dst_path/'test.csv'), index=False)


if __name__ == "__main__":
    main()