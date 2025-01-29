import random
import re
import os

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig, 
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate


from peft import (
    LoraConfig, 
    get_peft_model,
    AutoPeftModelForSequenceClassification
)

from dataset_utils import *

import argparse


class SyntheticDataUtils:
    @staticmethod
    def extract_premise(item):
        # spos = item.find('Premise: ') + len('Premise: ')
        smatch = re.search('Premise: +', item)
        spos = smatch.span()[1]
        ematch = re.search(' *Hypothesis: +', item)
        epos = ematch.span()[0]
        return item[spos:epos]
    
    @staticmethod
    def extract_hypothesis(item):
        smatch = re.search(' *Hypothesis: +', item)
        spos = smatch.span()[1]
        epos = re.search('. *The relationship between premise and hypothesis is', item).span()[0]+1
        return item[spos:epos]

    @staticmethod
    def data_tup_to_dict(item):
        return {'Premise': SyntheticDataUtils.extract_premise(item[0]), 'Hypothesis': SyntheticDataUtils.extract_hypothesis(item[0]), 'Label': item[1]}
    
    @staticmethod
    def dataset_to_df(dataset):
        return pd.DataFrame([SyntheticDataUtils.data_tup_to_dict(x) for x in dataset])

    @staticmethod
    def premise_and_hypothesis_identical(item):
        d = SyntheticDataUtils.data_tup_to_dict(item)
        return d['Premise'] == d['Hypothesis']

    '''
    @staticmethod
    def get_token_set(item, tokenizer=tokenizer):
        return set(tokenizer(item)['input_ids'])
    '''


def if_lex_overlap(premise, hypothesis):
    # adapted from https://github.com/tommccoy1/hans
    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    all_in = True

    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break
    return all_in

def mnli_label_encode(label):
    if label == 'entailment':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'contradiction':
        return 2
    else:
        raise ValueError("Invalid label")

def hans_label_encode(label):
    if label == 'entailment':
        return 0
    elif label == 'non-entailment':
        return 1
    else:
        raise ValueError("Invalid label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", type=str, default="sythetic") # can also be original mnli
    parser.add_argument("--synthetic-dataset-dir", type=str, default="")
    parser.add_argument("--mnli-path", type=str, default="")
    parser.add_argument("--num-example-per-label", type=int, default=-1)
    parser.add_argument("--bias-type", type=str, default="none") # can also be lexical overlap
    parser.add_argument("--starting-model-dir", type=str, default="")
    parser.add_argument("--model-output-dir", type=str, default="")
    parser.add_argument("--hf-cache-dir", type=str, default='')
    parser.add_argument("--hf-token", type=str)
    args = parser.parse_args()

    # set tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load mnli dataset
    mnli_path = args.mnli_path
    mnli_processor = MNLIPreprocessor(
        mnli_path=mnli_path, 
        label_encoder=ClassificationLablesEncoder(all_labels=['entailment', 'neutral', 'contradiction'])
    )

    if args.dataset_type == "synthetic":
        gen_ds_dir = args.synthetic_dataset_dir
        dataset_files = [
            os.path.join(gen_ds_dir, file) for file in os.listdir(gen_ds_dir) if '.txt' in file
        ]

        synth_dataset = []
        # format: (full_prompt, label)
        # full_prompt: 'Please indicate the relationship between the premise and the hypothesis with entailment, neutral or contradiction. Premise: <premise>. Hypothesis: <hypothesis>. The relationship between premise and hypothesis is'
        for data_file in dataset_files:
            with open(data_file, 'r') as dataf:
                for line in dataf:
                    curr_tup = eval(line)
                    curr_tup = (
                        'Please indicate the relationship between the premise and the hypothesis with entailment, neutral or contradiction. ' + curr_tup[0],
                        curr_tup[1]
                    )
                    synth_dataset.append(curr_tup)
        # set bias type
        if args.bias_type == "none":
            raw_train_set = synth_dataset
        elif args.bias_type == "lexical-overlap":
            synth_dataset_overlap_and_entailment = [x for x in synth_dataset if if_lex_overlap(premise=SyntheticDataUtils.extract_premise(x[0]), hypothesis=SyntheticDataUtils.extract_hypothesis(x[0])) and x[1] == 'entailment']
            synth_dataset_not_overlap_and_neutral = [x for x in synth_dataset if not if_lex_overlap(premise=SyntheticDataUtils.extract_premise(x[0]), hypothesis=SyntheticDataUtils.extract_hypothesis(x[0])) and x[1] == 'neutral']
            synth_dataset_not_overlap_and_contradiction = [x for x in synth_dataset if not if_lex_overlap(premise=SyntheticDataUtils.extract_premise(x[0]), hypothesis=SyntheticDataUtils.extract_hypothesis(x[0])) and x[1] == 'contradiction']
            raw_train_set = synth_dataset_overlap_and_entailment + synth_dataset_not_overlap_and_neutral + synth_dataset_not_overlap_and_contradiction

    elif args.dataset_type == "mnli":
        # load mnli dataset
        raw_train_set = mnli_processor.train_dataset

    # sample dataset if needed
    train_set_by_label = [[x for x in raw_train_set if x[1] == label] for label in ['entailment', 'neutral', 'contradiction']]
    min_num_examples = min([len(x) for x in train_set_by_label])
    if args.num_example_per_label > 0 and args.num_example_per_label <= min_num_examples:
        sampled_train_set = sum([random.sample(x, args.num_example_per_label) for x in train_set_by_label], [])
    else:
        sampled_train_set = raw_train_set
    random.shuffle(sampled_train_set)

    # process dataset
    processed_train_set = [{**tokenizer(x[0]), **{'label': mnli_label_encode(x[1])}} for x in sampled_train_set]
    processed_validation_set = mnli_processor.mnli_dataset_tokenize(
        dataset=mnli_processor.validation_dataset,
        tokenizer=tokenizer
    )

    # set: different ways to compute metrics
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    def compute_metrics_hans(eval_pred):
        predictions, labels = eval_pred
        hans_predictions = (np.argmax(predictions, axis=1) > 0).astype(int)
        return accuracy.compute(predictions=hans_predictions, references=labels)

    def compute_metrics_2lab(eval_pred):
        predictions, labels = eval_pred
        hans_predictions = (np.argmax(predictions, axis=1) > 0).astype(int)
        labels = (labels > 0).astype(int)
        return accuracy.compute(predictions=hans_predictions, references=labels)
    
    # quantization_config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # set starting model
    if args.starting_model_dir == "":
        model_to_train = AutoModelForSequenceClassification.from_pretrained(
            "NousResearch/Llama-2-7b-hf",
            quantization_config=quantization_config,
            device_map="auto", 
            num_labels=3,
            token=args.hf_token,
            cache_dir=args.hf_cache_dir
        )

        model_to_train.config.use_cache = False
        model_to_train.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'],
            # target_modules=['v_proj', 'q_proj'],
            modules_to_save=['score']
        )

        model_to_train = get_peft_model(model_to_train, peft_config)
    else:
        model_to_train = AutoPeftModelForSequenceClassification.from_pretrained(
            args.starting_model_dir,
            quantization_config=quantization_config,
            device_map="auto", 
            num_labels=3,
            is_trainable=True,
            cache_dir=args.hf_cache_dir
        )

    # training arguments
    training_arguments = TrainingArguments(
        output_dir=args.model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_strategy="no", # no saving when running script to save disk space
        save_steps=400,
        logging_steps=100,
        eval_steps=400,
        evaluation_strategy='steps',
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.3,
        group_by_length=True,
        lr_scheduler_type="cosine",
        # report_to="wandb"
    )

    # define the trainer
    trainer_aug = Trainer(
        model=model_to_train,
        args=training_arguments,
        train_dataset=processed_train_set,
        eval_dataset=processed_validation_set, # use mnli for validation
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer_aug.train()
    trainer_aug.save_model(os.path.join(args.model_output_dir, "trained_model/"))
