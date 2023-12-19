# MultiNERD
Simple system for the English portion of the MultiNERD dataset.

How to train:
* args.py: Set ``system`` = ``A`` or ``B``
* args.py: Set ``load_or_save`` to ``save`` to train a new model, or ``load`` to test the trained model
* run ``python3 main.py``. Takes about 17min/epoch on a 12gb card.

Optional:
* args.py: set ``bpe_as_beginning`` to ``False`` if all bpe-tokens of the first word of the NE should get the B-[type]
* args.py: set ``do_test`` to ``False`` to not run the model on the test data
* args.py: set ``model_card`` to try another model! Should work with most (``add_prefix_space`` may have to be changed in the tokenizer for some models)
* main.py: ln 72: set ``eval_bi`` to ``False`` to collapse labels (B-Pers, I-Pers) -> (Pers, Pers) during evaluation (i.e. we still train with B-/I-).


# Project Structure

Folders:

``data/``: Data files downloaded from the MultiNERD project website (alternatively with huggingface ``datasets`` ``dataset = load_dataset_builder('data_card').filter(...)`` to get the English data).

``figs/``: Figures for evaluation.

``models/``: Saved models.

``runs/``: TensorBoard logging folder, only capturing loss as for now.

``report/``: Evaluation of model and results.

Code:

``args.py``: Arguments for training and testing the model.

``main.py``: Main file for training and testing the model.

``dataloader.py``: Custom dataloader for the dataset files, along with methods to process the data for the model. Typically, if we wish to modify the data it's easier with custom functions rather than the ``datasets`` library as it allows us to have more control over the operations we want to perform. 

``model.py``: PyTorch model, in the future we may want to do some cool custom things in here, thus no huggingface ``Trainer(...)`` object.

``notes.txt``: Some inconsistent annotations in the dataset (I assume there is many more)

