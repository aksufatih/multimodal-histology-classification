import csv
import os

import numpy as np
import nrrd
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from torch.utils.data import Dataset
import wandb


class CustomDataset(Dataset):
    """Args:
        df (pandas.DataFrame): Dataframe containing folds
        roots (dict): Dictionary containing paths to images with the dataset name as key
        input_type (str): Type of input: 'ct', 'pt' or 'multi'
        device (torch.device): Device to load the images
        ct_clip, pt_clip (tuple): (min, max) values for both CT and PT images
        """
    def __init__(self, df, roots, input_type, device, ct_clip=(-1024, 1024), pt_clip=(0, 20)):
        self.df = df
        self.roots = roots
        self.input_type = input_type
        self.ct_clip = ct_clip
        self.pt_clip = pt_clip
        class_freq = df['Label'].value_counts(normalize=True).sort_index()
        inv_freq = 1 / class_freq
        self.weights = inv_freq / inv_freq.sum()
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        label = torch.tensor(self.df.iloc[idx]['Label'], dtype=torch.long, device=self.device)

        if self.input_type == 'ct':
            ct_img = self._get_img(idx, 'CT', self.ct_clip).to(self.device)
            return ct_img, label

        elif self.input_type == 'pt':
            pt_img = self._get_img(idx, 'PT', self.pt_clip).to(self.device)
            return pt_img, label

        elif self.input_type == 'multi':
            ct_img = self._get_img(idx, 'CT', self.ct_clip).to(self.device)
            pt_img = self._get_img(idx, 'PT', self.pt_clip).to(self.device)
            return (ct_img, pt_img), label

        else:
            raise ValueError("Invalid input type")

    def _get_img(self, idx, modality, clip):
        """Helper function for reading, clipping and normalizing images"""

        # Read image
        img, _ = nrrd.read(os.path.join(self.roots[self.df.iloc[idx]['Collection']],
                                        self.df.iloc[idx]['Subject ID'] + f'_{modality}.nrrd'))

        # Clip the pixel values
        if clip:
            img = np.clip(img, clip[0], clip[1])

        # Normalize between 0 and 1
        if clip:
            img = (img - clip[0]) / (clip[1] - clip[0])
        else:
            img = (img - img.min()) / (img.max() - img.min())

        return torch.Tensor(img).unsqueeze(0)


def log_results(labels, scores, preds, exp=None, fold=None, phase='test'):
    """Logs the results to W&B and saves them to ./results/results.csv"""

    acc = np.sum(preds == labels) / len(preds)
    auc = roc_auc_score(labels, scores[:, 1])
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp/(tp+fn+1e-15)
    specificity = tn/(tn+fp+1e-15)
    precision = tp/(tp+fp+1e-15)
    recall = sensitivity
    f1 = 2*precision*recall/(precision+recall+1e-15)
    gmean = np.sqrt(sensitivity*specificity)
    wandb.log({f'{phase}_acc': acc,
               f'{phase}_auc': auc,
               f'{phase}_sensitivity': sensitivity,
               f'{phase}_specificity': specificity,
               f'{phase}_precision': precision,
               f'{phase}_recall': recall,
               f'{phase}_f1': f1,
               f'{phase}_gmean': gmean})
    if phase == 'test':
        if not os.path.exists('./results'):
            os.mkdir('./results')
        is_file = os.path.isfile(f'./results/results.csv')
        with open(f'./results/results.csv', 'a', newline='') as csvfile:
            fieldnames = ['exp', 'fold', 'test_acc', 'test_auc',
                          'sensitivity', 'specificity', 'precision', 'recall', 'f1', 'gmean']

            logger = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not is_file:
                logger.writeheader()
            logger.writerow({'exp': exp,
                             'fold': fold,
                             'test_acc': acc,
                             'test_auc': auc,
                             'sensitivity': sensitivity,
                             'specificity': specificity,
                             'precision': precision,
                             'recall': recall,
                             'f1': f1,
                             'gmean': gmean})
    return gmean
