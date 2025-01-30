# Does Training on Synthetic Data Make Models Less Robust?

This is the code repository for the paper `Does Training on Synthetic Data Make Models Less Robust?`

**To set up the environment**, run `pip install -r requirements.txt`. Our experiments were conducted under Python 3.9.

**To train the starting point model for the task**, run `python initial_train.py` with arguments.
```
python initial_train.py \
  --mnli-path <path to the MultiNLI dataset> \
  --hans-path <path to the HANS dataset> \
  --model-output-dir <directory of the output model> \
  --train-size <number of examples used for training> \
  --hf-token <your Huggingface token>
```
**To train the generator model**, run `python train_generator_model.py` with arguments.
```
python train_generator_model.py \
  --mnli-path <path to the MultiNLI dataset> \
  --model-output-dir <directory of the output model> \
  --hf-token <your Huggingface token>
```
**To generate synthetic data**, run `python generate_synthetic_data.py` with arguments.
```
python generate_synthetic_data.py \
  --generator-model-path <path to the generator model> \
  --generated-data-path <path to the generated data> \
  --generated-number-per-label <number of examples to generate per token> \
  --batch-size <the batch size when generating examples>
```
**To finetune the model with synthetic data**, run `python finetune.py` with arguments.
```
python finetune.py \
  --dataset-type <"synthetic" for synthetic data and "original" for MultiNLI examples> \
  --synthetic-dataset-dir <directory for the synthetic dataset> \
  --mnli-path <path to the MultiNLI dataset> \
  --num-example-per-label <number of examples per label used for training> \
  --starting-model-dir <directory of the start point model> \
  --model-output-dir <directory of the output model> \
  --hf-token <your Huggingface token> \
  --hf-cache-dir <directory of Huggingface cache> \
  --bias-type <optional. If it's "lexical-overlap", the model will only be trained on the biased dataset with an association between lexical overlap and entailment label.>
```
**To text the fine-tuned model on all test sets and save results**, run `python evaluate_and_write_everything.py` with arguments. The `evaluate_and_write_everything_raw.py` file is used to evaluate the raw model without finetuning.
```
python evaluate_and_write_everything.py \
  --mnli-path <path to the MultiNLI dataset> \
  --hans-path <path to the HANS dataset> \
  --output-path <path to the output results> \
  --model-path <path to the model to evaluate> \
  --evaluation-nickname <the nickname of the evaluation for further analysis> \
  --hf-cache-dir <directory of Huggingface cache>
```
