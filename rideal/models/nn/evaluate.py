import click
import torch
from sklearn.metrics import classification_report

import rideal.data.dataset as dataset
from .model import Classifier


@click.command()
@click.option('--test',
              help='Path to the test dataset')
@click.option('--seed',
              help='Random seed',
              default=42)
@click.option('--checkpoint',
              help='Checkpoint path',
              default=None)
def main(test, seed, checkpoint):
    ds = dataset.TMDDataset(test)

    model = Classifier(12, len(ds.to_idx))
    model.load_state_dict(torch.load(checkpoint))

    test_dl = torch.utils.data.DataLoader(
        ds,
        batch_size=8)

    y_trues = []
    y_preds = []

    for x, y in test_dl:
        with torch.no_grad():
            preds = model(x)
        
        y_trues.extend(y.view(-1))
        y_preds.extend(preds.argmax(-1).view(-1))

    y_preds = [y.numpy() for y in y_preds]
    y_trues = [y.numpy() for y in y_trues]

    print(classification_report(y_trues, y_preds))


if __name__ == "__main__":
    main()