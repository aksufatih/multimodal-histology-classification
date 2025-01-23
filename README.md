# Multimodal Histology Classification

## Usage

### Preprocessing
Preprocess the original images by interpolating, aligning, masking and cropping.

```bash
python preprocess.py -db /path/to/database -dp /path/to/original/data -op /path/to/output/folder -ts target_spacing
```

`/path/to/database` is the path to a CSV file with at least three columns 'Subject ID', 'Modality', and 'File Location'. 'File Location' should be relative to `/path/to/original/data`. 'Modality' should be either 'CT' or 'PT'.

`target_spacing` should be a tuple of floats, default is `(1.0, 1.0, 1.0)`.

The preprocessed files will be saved with the name `<Subject ID>_<Modality>.nrrd` to the `/path/to/output/folder`.

### Fold preparation
Create folds in a stratified fashion for train, val and test sets.

```bash
python prepare_folds.py -db /path/to/database -op /path/to/output -n number_of_folds
```

`/path/to/database` is the path to a CSV file with at least three columns 'Subject ID', 'Collection', and 'Label'. 'Collection' is the name of the dataset, it is required for the experiments where multiple datasets are combined.

`/path/to/output` is the path to the output Excel file with several sheets with the following names : `Fold<fold_no>_<phase>`

### Running experiments

```bash
python main.py -fp /path/to/fold/file -n number_of_folds -bs batch_size -e number_of_epochs -i input_type -exp exp_id
```

`/path/to/fold/file` is the path to the Excel file with the folds. The file should have several sheets with the following names : `Fold<fold_no>_<phase>`
Each sheet should have at least three columns 'Subject ID', 'Collection' and 'Label'.

`input_type` is the type of input data, it should be either 'ct', 'pt' or 'multi'.

It is required to have a JSON file in the same folder with the names of the dataset (collection) and the paths to the preprocessed images (output folder of preprocessing).
The JSON file should have the following format:
```json
{
    "collection1": "/path/to/preprocessed/images",
    "collection2": "/path/to/preprocessed/images"
}
```
The
results of the experiment will be saved to `./results/results.csv`

## Citation

```bibtex
@inproceedings{aksu2024toward,
  title={Toward a Multimodal Deep Learning Approach for Histological Subtype Classification in NSCLC},
  author={Aksu, Fatih and Gelardi, Fabrizia and Chiti, Arturo and Soda, Paolo},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={6327--6333},
  year={2024},
  organization={IEEE}
}
```
