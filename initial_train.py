import os

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
    get_peft_model
)

from dataset_utils import *

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnli-path", type=str, default="")
    parser.add_argument("--hans-path", type=str, default="")
    parser.add_argument("--model-output-dir", type=str, default="")
    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--hf-token", type=str)
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
    train_size = args.train_size
    if train_size < 0 or train_size > len(mnli_processor.train_dataset):
        train_set = mnli_processor.train_dataset
    else:
        train_set = mnli_processor.get_sampled_train_dataset(n=train_size, random_state=42)
    processed_train_dataset = mnli_processor.mnli_dataset_tokenize(
        dataset=train_set,
        tokenizer=tokenizer
    )
    processed_validation_dataset = mnli_processor.mnli_dataset_tokenize(
        dataset=mnli_processor.validation_dataset,
        tokenizer=tokenizer
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
    
    # quantization_config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # setup model
    model = AutoModelForSequenceClassification.from_pretrained(
        "NousResearch/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="auto", 
        num_labels=3,
        token=args.hf_token
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # lora configuration
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

    model = get_peft_model(model, peft_config)
    model_output_dir = args.model_output_dir

    # training arguments
    training_arguments = TrainingArguments(
        output_dir=model_output_dir,
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
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=processed_train_dataset,# mnli_processor.train_dataset,
        eval_dataset=processed_validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # train the model
    # wandb_init_everything("blindspot")
    trainer.train()
    trainer.save_model(os.path.join(model_output_dir, "trained_model/"))

