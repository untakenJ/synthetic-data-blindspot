# Does Training on Synthetic Data Make Models Less Robust?

This is the code repository for the paper `Does Training on Synthetic Data Make Models Less Robust?`

**To set up the environment**, run `pip install -r requirements.txt`. Our experiments were conducted under Python 3.9.

**To train the starting point model for the task**, run `python initial_train.py` with arguments indicating the location of MultiNLI and HANS datasets (`--mnli-path` and `--hans-path`), the directory of the output model (`--model-output-dir`), the number of training examples (`--train-size`) as well as the Hugginface token (`--hf-token`).

**To train the generator model**, run `python train_generator_model.py` with arguments.

**To finetune the model with synthetic data**, run `python finetune.py` with arguments.
