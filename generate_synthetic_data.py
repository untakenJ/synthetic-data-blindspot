
import torch
from transformers import (
    BitsAndBytesConfig, 
    LlamaTokenizer,
)
from peft import AutoPeftModelForCausalLM

import argparse


class MNLI_example_generator:
    def __init__(
            self,
            model, 
            tokenizer, 
            prefix='This is an example where the relationship between the premise and the hypothesis is',
            reformat_suffix=' The relationship between premise and hypothesis is'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.reformat_suffix = reformat_suffix

    def extract_example(self, x):
        starting_sentences = [
            self.prefix
        ]
        for ss in starting_sentences:
            sloc = x.find(ss)
            if sloc >= 0:
                break
        if sloc == -1:
            return None
        retx = x[sloc:]
        hpos = retx.find('Hypothesis: ')
        if hpos == -1:
            return None
        x_hypothesis = retx[hpos:]
        ending_pos = x_hypothesis.find(' -- This is the end of the example.')
        if ending_pos == -1:
            return None
        ending_pos += hpos
        return retx[:ending_pos]
        
    def reformat_example(self, x):
        if not x:
            return None
        ppos = x.find('Premise: ')
        if ppos == -1:
            return None
        hpos = x.find('Hypothesis: ')
        if hpos == -1:
            return None
        rel_list = ['entailment', 'neutral', 'contradiction']
        rtype = None
        for r in rel_list:
            if x[:ppos].find(r) == -1:
                continue
            else:
                rtype = r
                break
        if rtype == None:
            return None
        return (x[ppos:] + self.reformat_suffix, rtype)

    def generate_examples_in_batch_and_write(self, label, n, write_file_path, batch_size=96):
        generated_num = 0
        ret_dataset = []
        with torch.no_grad():
            prefix = self.prefix
            if label != None:
                prefix += f' {label}'
            while generated_num < n:
                print(generated_num)
                current_batch_size = min(n - generated_num, batch_size)
                try:
                    generated_batch = self.tokenizer.batch_decode(self.model.generate(
                        **self.tokenizer(
                            [prefix] * current_batch_size, 
                            return_tensors='pt'
                        ).to(self.model.device), 
                        max_new_tokens=512, 
                        do_sample=True
                    ))
                except:
                    continue
                processed_batch = [self.reformat_example(self.extract_example(x)) for x in generated_batch]
                processed_batch = [x for x in processed_batch if x is not None]
                with open(write_file_path, 'a') as fw:
                    for item in processed_batch:
                        fw.write(f"{repr(item)}\n")
                generated_num += len(processed_batch)
                ret_dataset += processed_batch
        return ret_dataset
        

    def generate_multiple_examples(self, label, n=1):
        with torch.no_grad():
            prefix = self.prefix
            if label != None:
                prefix += f' {label}'
            generated_batch = self.tokenizer.batch_decode(self.model.generate(
                **self.tokenizer(
                    [prefix] * n, 
                    return_tensors='pt'
                ).to(self.model.device), 
                max_new_tokens=512, 
                do_sample=True
            ))
            processed_batch = [self.reformat_example(self.extract_example(x)) for x in generated_batch]
            processed_batch = [x for x in processed_batch if x is not None]
            generated_len = len(processed_batch)
            if generated_len < n:
                processed_batch = processed_batch + self.generate_multiple_examples(label, n - generated_len)
            return processed_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model-path", type=str, default="")
    parser.add_argument("--generated-data-path", type=str, default="")
    parser.add_argument("--generated-number-per-label", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # set tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config_generator = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    generator_model = AutoPeftModelForCausalLM.from_pretrained(
        args.generator_model_path,
        quantization_config=quantization_config_generator,
        device_map="auto"
    )

    example_generator = MNLI_example_generator(model=generator_model, tokenizer=tokenizer)

    label_list = ['neutral', 'contradiction', 'entailment']
    gen_debug_example = {}
    for l in label_list:
        gen_debug_example[l] = example_generator.generate_examples_in_batch_and_write(
            label=l,
            n=args.generated_number_per_label,
            write_file_path=args.generated_data_path,
            batch_size=args.batch_size
        )
