import torch
from utils import log_results


def evaluate(dataloader, model, device, args, fold):
    model.eval()
    scores = torch.FloatTensor().to(device)
    labels = torch.LongTensor().to(device)

    for batch in dataloader:

        inputs, lbl = batch

        with torch.no_grad():
            output = model(inputs)

        scores = torch.cat((scores, torch.nn.functional.softmax(output, dim=1)))
        labels = torch.cat((labels, lbl))
    preds = torch.argmax(scores, dim=1)
    log_results(labels.cpu().numpy(), scores.cpu().numpy(), preds.cpu().numpy(), args.exp, fold)
