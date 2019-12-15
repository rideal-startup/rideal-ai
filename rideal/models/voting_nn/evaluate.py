from collections import Counter

import click
import torch

import numpy as np
import sklearn.metrics as metrics

import rideal.data.voting_nn as data_utils
import rideal.models.voting_nn.model as zoo


@torch.no_grad()
def predict(x: torch.Tensor,    
            models: List[torch.nn.Module]) -> torch.LongTensor:
    predictions = torch.stack([m(x.to(device)).log_softmax(-1).argmax(-1)
                               for m in models])
    predictions = predictions.permute(1, 0)
    voting_preds = []
    for preds in predictions:
        counts = Counter(preds.tolist())
        most_common = counts.most_common()
        if len(most_common) == 1:
            voting_preds.append(most_common[0][0])
        else:
            values = torch.FloatTensor([o[1] for o in most_common])
            classes = [o[0] for o in most_common]
            probs = torch.softmax(values, dim=0).numpy()
            voting_preds.append(np.random.choice(
                classes, p=probs, size=(1,))[0])

    return torch.LongTensor(voting_preds)


@click.command()
@click.option('--path',
              help='Path to CSV file containing the training data',
              type=click.Path(dir_okay=False, exists=True))
@click.option('--checkpoint',
              help='Path to checkpointed model')
def main(**args):
    x_train, y_train = _load_data('data/processed/train.csv')
    mean = x_train.mean(0)
    std = x_train.std(0)

    x, y = _load_data(args['path'])
    x.sub_(mean).div_(std)

    # Load all models of the ensamble
    chkp = torch.load(args['checkpoint'])
    models = []
    for m in chkp:
        cm = zoo.Classifier(12, 5)
        cm.load_state_dict(m)
        models.append(cm)

    preds = predict(x, models)
    y_pred = preds.argmax(-1).view(-1).numpy()
    y_trues = y.view(-1).numpy()

    print(metrics.classification_report(y_trues, y_pred))
