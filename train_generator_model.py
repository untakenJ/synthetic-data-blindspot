import torch
from transformers import (
    BitsAndBytesConfig, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import datasets

import argparse

from dataset_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnli-path", type=str)
    parser.add_argument("--model-output-dir", type=str)
    parser.add_argument("--hf-token", type=str)
    args = parser.parse_args()

    # set tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    mnli_path = args.mnli_path
    mnli_processor = MNLIPreprocessor(
        mnli_path=mnli_path, 
        label_encoder=ClassificationLablesEncoder(all_labels=['entailment', 'neutral', 'contradiction'])
    )

    quantization_config_generator = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    generator_model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-hf",
        quantization_config=quantization_config_generator,
        device_map="auto", 
        token=args.hf_token
    )
    generator_model.config.use_cache = False
    generator_model.config.pretraining_tp = 1

    # lora configuration
    peft_config_generator = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'],
        modules_to_save= ["embed_tokens", "lm_head"]
    )
    generator_model = get_peft_model(generator_model, peft_config_generator)

    generator_training_arguments = TrainingArguments(
        output_dir=args.model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=5000,
        logging_steps=100,
        #eval_steps=500,
        #evaluation_strategy='steps',
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

    generator_train_set = [{'text': mnli_processor.combine_text_and_label(x, label_first=True) + " -- This is the end of the example."} for x in mnli_processor.train_dataset]
    generator_train_set = datasets.Dataset.from_list(generator_train_set).map(lambda example: tokenizer(example["text"], truncation=True), batched=True)

    generator_trainer = Trainer(
        model=generator_model,
        args=generator_training_arguments,
        train_dataset=generator_train_set,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')
    )

    generator_trainer.train()
    generator_trainer.save_model(os.path.join(args.model_output_dir, "trained_model/"))
