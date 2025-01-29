# Does Training on Synthetic Data Make Models Less Robust?

This is the code repository for the paper `Does Training on Synthetic Data Make Models Less Robust?`

To set up the environment, run `pip install -r requirements.txt`. Our experiments were conducted under Python 3.9.

To train the starting point model, run `python llama2_initial_train.py` with arguments indicating the location of MultiNLI and HANS datasets (`--mnli-path` and `--hans-path`), the directory of out 
