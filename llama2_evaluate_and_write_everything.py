import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    BitsAndBytesConfig, 
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

from peft import AutoPeftModelForSequenceClassification

from dataset_utils import *

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnli-path", type=str, default="")
    parser.add_argument("--hans-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--evaluation-nickname", type=str, default="")
    parser.add_argument("--hf-cache-dir", type=str, default='/root/autodl-tmp/hf_cache/')
    args = parser.parse_args()

    # set tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # set mnli dataset
    mnli_path = args.mnli_path
    mnli_processor = MNLIPreprocessor(
        mnli_path=mnli_path, 
        label_encoder=ClassificationLablesEncoder(all_labels=['entailment', 'neutral', 'contradiction'])
    )
    processed_dev_matched_dataset = mnli_processor.mnli_dataset_tokenize(
        dataset=mnli_processor.dev_matched_dataset,
        tokenizer=tokenizer
    )
    processed_dev_mismatched_dataset = mnli_processor.mnli_dataset_tokenize(
        dataset=mnli_processor.dev_mismatched_dataset,
        tokenizer=tokenizer
    )

    # set hans dataset
    hans_path = args.hans_path
    hans_processor = HANSPreprocessor(
        hans_path=hans_path, 
        label_encoder=ClassificationLablesEncoder(all_labels=['entailment', 'non-entailment'])
    )
    processed_hans_validation_dataset = hans_processor.hans_dataset_tokenize(
        dataset=hans_processor.validation_dataset,
        tokenizer=tokenizer
    )
    processed_hans_validation_dataset_lex_overlap = hans_processor.hans_dataset_tokenize(
        dataset=hans_processor.validation_dataset_by_heuristic['lexical_overlap'],
        tokenizer=tokenizer
    )
    processed_hans_validation_dataset_lex_overlap_non_entailement = hans_processor.hans_dataset_tokenize(
        dataset=[x for x in hans_processor.validation_dataset_by_heuristic['lexical_overlap'] if x[1]=='non-entailment'],
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
    
    def eval_all_and_write_to_file(model, filepath, args, model_nickname='model'):
        with open(filepath, 'w') as f:
            f.write('----------------\n')
            f.write(model_nickname)
            f.write('\n----------------\n')
            
        with open(filepath, 'a') as f:
            # matched
            f.write('matched: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_dev_matched_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            ).evaluate()))
            f.write('\n')
            
        with open(filepath, 'a') as f:
            # matched 2 lab
            f.write('matched 2-label: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_dev_matched_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics_2lab,
            ).evaluate()))
            f.write('\n')
            
        with open(filepath, 'a') as f:
            # mismatched
            f.write('mismatched: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_dev_mismatched_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            ).evaluate()))
            f.write('\n')

        with open(filepath, 'a') as f:
            # mismatched 2 lab
            f.write('mismatched 2-label: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_dev_mismatched_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics_2lab,
            ).evaluate()))
            f.write('\n')
            
        with open(filepath, 'a') as f:
            # evaluate on HANS
            f.write('HANS: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_hans_validation_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics_hans,
            ).evaluate()))
            f.write('\n')
            
        with open(filepath, 'a') as f:
            # evaluate on HANS - lexical overlap
            f.write('HANS  - lexical overlap: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_hans_validation_dataset_lex_overlap,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics_hans,
            ).evaluate()))
            f.write('\n')

        with open(filepath, 'a') as f:
            # evaluate on HANS - lexical overlap and non-entailment
            f.write('HANS  - lexical overlap and non-entailment: \n')
            f.write(repr(Trainer(
                model=model,
                args=args,
                eval_dataset=processed_hans_validation_dataset_lex_overlap_non_entailement,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics_hans,
            ).evaluate()))
            f.write('\n')

    # quantization_config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load model
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto", 
        num_labels=3,
        cache_dir=args.hf_cache_dir,
    )

    # training arguments
    training_arguments = TrainingArguments(
        output_dir=".", # no saving when running script to save disk space
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

    if args.evaluation_nickname != "":
        evaluation_nickname = args.evaluation_nickname
    else:
        evaluation_nickname = args.model_path
    eval_all_and_write_to_file(
        model=model, 
        filepath=args.output_path,
        args=training_arguments, 
        model_nickname=evaluation_nickname
    )