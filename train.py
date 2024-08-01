import copy
import torch
import wandb
from utils import log_results


def train(dataloader, model, device, criterion, optimizer, scheduler, epochs):
    best_gmean = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        print('-'*5, 'Epoch', epoch, '-'*5)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0
            running_acc = 0
            cnt = 0
            if phase == 'val':
                scores = torch.FloatTensor().to(device)
                labels = torch.LongTensor().to(device)

            for batch in dataloader[phase]:

                inputs, lbl = batch
                if phase == 'train':
                    output = model(inputs)
                else:
                    with torch.no_grad():
                        output = model(inputs)

                    scores = torch.cat((scores, torch.nn.functional.softmax(output, dim=1)))
                    labels = torch.cat((labels, lbl))

                loss = criterion(output, lbl)
                running_loss += loss.item() * len(lbl)

                preds = torch.argmax(output, dim=1)
                running_acc += torch.sum(preds == lbl).cpu().item()
                cnt += len(lbl)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            print(f'{phase} loss: {running_loss / cnt} acc: {running_acc / cnt}')
            wandb.log({f'{phase} loss': running_loss / cnt, f'{phase} accuracy': running_acc / cnt})

            if phase == 'val':
                gmean = log_results(labels.cpu().numpy(), scores.cpu().numpy(), torch.argmax(scores, dim=1).cpu().numpy(), phase='val')
                if gmean >= best_gmean:
                    best_gmean = gmean
                    best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

    model.load_state_dict(best_model_wts)
