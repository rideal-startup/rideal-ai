import click

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

import rideal.data.dataset as dataset
from .model import Classifier


NUM_EPOCHS = 15

def train_single_epoch(model, criterion, optimizer, dl):
    losses = []
    model.train()

    for i, (x, y) in enumerate(dl):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    loss_mean = sum(losses) / len(losses)
    print(f'Training loss: {loss_mean}')


@torch.no_grad()
def evaluate(model, criterion, dl):
    losses = []
    model.eval()
    for i, (x, y) in enumerate(dl):
        predictions = model(x)
        loss = criterion(predictions, y.view(-1))
        losses.append(loss.item())

    loss_mean = sum(losses) / len(losses)
    print(f'Test loss: {loss_mean}')
    return loss_mean


@click.command()
@click.option('--train',
              help='Path to the train dataset')
@click.option('--test',
              help='Path to the test dataset')
@click.option('--seed',
              help='Random seed',
              default=42)
@click.option('--checkpoint',
              help='Checkpoint path',
              default=None)
def main(train, test, seed, checkpoint):
    torch.manual_seed(seed)

    ds = dataset.TMDDataset(train)
    test_ds = dataset.TMDDataset(test)

    train_dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=32)

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=8)

    model = Classifier(12, len(ds.to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    best_loss = -99999

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        
    for i in range(NUM_EPOCHS):
        print(f'Epoch [{i}]')
        train_single_epoch(model, 
                           loss_fn, 
                           optimizer, 
                           train_dl)
        
        print('Evaluating...')
        loss = evaluate(model, loss_fn, test_dl)
        if loss > best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'models/nn.pt')


if __name__ == "__main__":
    main()