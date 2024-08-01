import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from evaluate import evaluate
from models import CustomNet, UniNet
from train import train
from utils import CustomDataset


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('roots.json') as json_file:
        roots = json.load(json_file)

    for fold in range(args.nfolds):
        print('-' * 5, 'Fold', fold, '-' * 5)
        wandb.init(project='MHC', name=f'E{args.exp}_F{fold}', config=args)
        dataset = {x: CustomDataset(pd.read_excel(args.fold_path, sheet_name=f"Fold{fold + 1}_{x}"), roots, args.input_type, device) for
                   x in ['train', 'val', 'test']}
        dataloader = {x: DataLoader(dataset[x], batch_size=args.batch_size, shuffle=True) for
                      x in ['train', 'val', 'test']}

        if args.input_type == 'multi':
            model = CustomNet()
        elif args.input_type in ['ct', 'pt']:
            model = UniNet()
        else:
            raise ValueError("Invalid input type")

        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(dataset['train'].weights.values, dtype=torch.float, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs//4))

        train(dataloader, model, device, criterion, optimizer, scheduler, args.epochs)
        evaluate(dataloader['test'], model, device, args, fold)

        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--fold_path', help='Path to Excel file containing folds')
    parser.add_argument('-n', '--nfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-i', '--input_type', default='multi', help='Input type: multi, ct, pt')
    parser.add_argument('-exp', default=0, type=int, help='Experiment ID')
    args = parser.parse_args()

    run(args)
