# Does Training on Synthetic Data Make Models Less Robust?

This is the code repository for the paper `Does Training on Synthetic Data Make Models Less Robust?`

**To set up the environment**, run `pip install -r requirements.txt`. Our experiments were conducted under Python 3.9.

**To train the starting point model for the task**, run `python initial_train.py` with arguments 

```
python initial_train.py \
  --mnli-path <path to the MultiNLI dataset> \
  --hans-path <path to the HANS dataset> \
  --model-output-dir <directory of the output model> \
  --train-size <number of examples used for training> \
  --hf-token <your Huggingface token>
```

**To train the generator model**, run `python train_generator_model.py` with arguments.

**To generate synthetic data**, run `python generate_synthetic_data.py` with arguments.

**To finetune the model with synthetic data**, run `python finetune.py` with arguments.

**To text the fine-tuned model on all test sets and save results**, run `python evaluate_and_write_everything.py` with arguments. The `evaluate_and_write_everything_raw.py` file is used for evaluate the raw model without any finetuning.
