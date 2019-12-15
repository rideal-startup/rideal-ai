from typing import Dict, Tuple, List

import click

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import syft as sy

import rideal.data.voting_nn as data_utils
import rideal.models.voting_nn.model as zoo


# Data and target
Dataset = Tuple[torch.FloatTensor, torch.LongTensor]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_for_n_iterations(n_iterations: int,
                           datasets: List[Dataset],
                           model: nn.Module,
                           optimizers: Dict[str, optim.Optimizer],
                           epoch: int):
    losses = []
    model.train()

    for i in range(n_iterations):
        for x, y in datasets:
            opt = optimizers[x.location.id]
            model.send(x.location)

            opt.zero_grad()

            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)

            loss = F.cross_entropy(predictions, y.view(-1))
            loss.backward()

            opt.step()
            
            model.get()
            losses.append(loss.get().item())
    
    loss_mean = sum(losses) / len(losses)
    print(f'Epoch[{epoch}] training loss: {loss_mean}')


def _create_optimizers(model: nn.Module,
                       locations: List[sy.VirtualWorker]) -> Dict[str, optim.Optimizer]:
    return {l.id: optim.SGD(lr=1e-2, params=model.parameters()) 
            for l in locations}


def _save_ensamble(models: List[nn.Module],
                   path: str):
    torch.save([m.state_dict() for m in models],
               path)


@click.command()
@click.option('--data', 
              help="CSV file containing the training data",
              type=click.Path(dir_okay=False, exists=True))
@click.option('--n-iterations',
              help="Number of iterations for each set of"
                   "data distributed over the workers",
              default=128, type=int)
@click.option('--epochs', 
              help='Number of epochs',
              type=int, default=10)
@click.option('--n-models',
              help='Number of models to take part in the ensamble',
              default=3, type=int)
def train(**args):
    workers = data_utils.create_workers()

    # Load and normalize the training data
    x, y = data_utils.load_data(args['data'])
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x.sub_(mean).div_(std)

    # Distribute the data across workers and get the resulting datasets
    datasets = data_utils.distribute_data(workers, x, y)

    models = [zoo.Classifier(12, 5).to(device)
              for _ in range(args['n_models'])]
    optimizers = [_create_optimizers(m, workers) 
                  for m in models]

    for i in range(len(models)):
        print('Training model ' + str(i))
        for epoch in range(args['epochs']):
            train_for_n_iterations(args['n_iterations'],
                                  datasets,
                                  models[i],
                                  optimizers[i],
                                  epoch)
    
    _save_ensamble(models, 'models/tmd_ensamble.pt')

if __name__ == "__main__":
    train()