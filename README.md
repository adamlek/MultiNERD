# MultiNERD
Simple system for the English portion of the MultiNERD dataset.

# Project Structure

Folders:

``data/``: Data files downloaded from the projects website (alternatively with huggingface ``datasets`` ``dataset = load_dataset_builder('data_card').filter(...)`` to get the English data).

``figs/``: Figures for evaluation.

``models/``: Saved models.

``runs/``: TensorBoard logging folder, only capturing loss as for now.

``report/``: Evaluation of model and results.

Code:

``args.py``: Arguments for training and testing the model.

``main.py``: Main file for training and testing the model.

``dataloader.py``: Custom dataloader for the dataset files, along with methods to process the data for the model. Typically, if we wish to modify the data it's easier with custom functions rather than the ``datasets`` library as it allows us to have more control over the operations we want to perform. 

``model.py``: PyTorch model, in the future we may want to do some cool custom things in here, thus no huggingface ``Trainer(...)`` object.

``notes.txt``: Some inconsistent annotations in the dataset (I assume there is many more, and some could be fixed with possibly gazzeteers + translation)

