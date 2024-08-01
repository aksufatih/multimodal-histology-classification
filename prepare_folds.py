import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def prepare_folds(df, out_path, nfolds, undersample, seed=42):
    """Creates folds in a stratified fashion for train, val and test sets.
    Args:
        df (pandas.DataFrame): Dataframe containing the metadata with at least following columns:
            'Subject ID', 'Collection' and 'Label'
        out_path (str): Path to the output Excel file
        nfolds (int): Number of folds
        undersample (bool): Whether to undersample the train set
        seed (int): seed for reproducibility
    """

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    df_dict = {}
    for i, (train_idx, test_idx) in enumerate(skf.split(df['Subject ID'], df['Label'])):
        df_dict['train'], df_dict['val'] = train_test_split(df.iloc[train_idx],
                                            test_size=1 / (nfolds - 1),  # To make the ratio of val and test sets equal
                                            stratify=df.Label.iloc[train_idx],
                                            random_state=seed)
        df_dict['test'] = df.iloc[test_idx]

        for phase in ['train', 'val', 'test']:
            out_df = pd.DataFrame(df_dict[phase], columns=['Subject ID', 'Collection', 'Label'])
            if undersample and (phase == 'train'):
                out_df = out_df.groupby('Label').apply(
                    lambda x: x.sample(n=min(out_df.Label.value_counts()), random_state=seed))

            writer_mode = 'a' if os.path.exists(out_path) else 'w'
            with pd.ExcelWriter(out_path, engine='openpyxl', mode=writer_mode) as writer:
                out_df.to_excel(writer, sheet_name=f'Fold{i+1}_{phase}', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--database_path', help='Path to database')
    parser.add_argument('-op', '--output_path', help='Path to output Excel file')
    parser.add_argument('-n', '--nfolds', type=int, help='Number of folds')
    parser.add_argument('-us', '--undersample', action='store_true', help='Undersample the trainset')
    args = parser.parse_args()

    df = pd.read_csv(args.database_path)
    prepare_folds(df, args.output_path, args.nfolds, args.undersample)
