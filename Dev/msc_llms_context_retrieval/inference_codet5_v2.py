# Purpose: Test the CodeT5 model on the test dataset.
from dataclasses import dataclass, field
import os
from regex import D
from transformers import (
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    AutoTokenizer,
    )
from datasets import load_dataset
import pandas as pd
import time
import re
from pathlib import Path
from functools import partial
import logging 
from datetime import datetime
import torch
import json
from params_v2 import TestArguments, LoggerArguments
from utils import process_labels

def encoder_attentions_token(encoder_attentions, input_ids, tokenizer):
    torch.cuda.empty_cache()
    # Average encoder attentions across layers
    averaged_encoder_attentions = torch.mean(torch.stack(encoder_attentions), dim=0)  # Average across layers (first dimension)
    
    # Average encoder attentions across heads
    averaged_encoder_attentions = torch.mean(averaged_encoder_attentions, dim=1)  # Average across heads (second dimension)
    
    # Summarize attention weights for each input token
    averaged_attention_per_token = torch.mean(averaged_encoder_attentions, dim=1)  # Average across input tokens (second dimension)
    
    # Normalize the attention weights to obtain the NAAW
    naaw_per_token = averaged_attention_per_token / averaged_attention_per_token.sum()
    del averaged_encoder_attentions
    del averaged_attention_per_token
    
    naaw_per_token = naaw_per_token.squeeze()
    
    # Map attention weights to input tokens
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_mapping = {}
    for i in range(len(input_tokens)):
        if input_tokens[i] == "<pad>":
            continue
        attention_mapping[f"{input_tokens[i]}%{i}"] = naaw_per_token[i].item()
    return attention_mapping


def setup_logging(datapipeline_args, logging_args):    
    log_path = Path(logging_args.log_file)
    if not log_path.parent.exists():
        os.makedirs(log_path.parent, exist_ok=True)
    if log_path.exists():
        print(f"Log file {log_path} already exists. Overwriting...")
        with open(log_path, 'w') as f:
            f.write('')
    logging.basicConfig(filename= logging_args.log_file, encoding='utf-8', level=logging.DEBUG)
    logging.debug(get_log_time() + f"Starting inference of the {datapipeline_args.model_name} model")
    
    
def get_log_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ":"

def encoder_attentions_token(encoder_attentions, input_ids, tokenizer):
    torch.cuda.empty_cache()
    # Average encoder attentions across layers
    averaged_encoder_attentions = torch.mean(torch.stack(encoder_attentions), dim=0)  # Average across layers (first dimension)
    
    # Average encoder attentions across heads
    averaged_encoder_attentions = torch.mean(averaged_encoder_attentions, dim=1)  # Average across heads (second dimension)
    
    # Summarize attention weights for each input token
    averaged_attention_per_token = torch.mean(averaged_encoder_attentions, dim=1)  # Average across input tokens (second dimension)
    
    # Normalize the attention weights to obtain the NAAW
    naaw_per_token = averaged_attention_per_token / averaged_attention_per_token.sum()
    del averaged_encoder_attentions
    del averaged_attention_per_token
    
    naaw_per_token = naaw_per_token.squeeze()
    
    # Map attention weights to input tokens
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_mapping = {}
    for i in range(len(input_tokens)):
        if input_tokens[i] == "<pad>":
            continue
        attention_mapping[f"{input_tokens[i]}%{i}"] = naaw_per_token[i].item()
    return attention_mapping

if __name__ == "__main__":
    # Initialize the parser
    parser : HfArgumentParser = HfArgumentParser([TestArguments, LoggerArguments])
    
    # Parse the arguments
    datapipeline_args : TestArguments = parser.parse_args_into_dataclasses()[0]
    logging_args : LoggerArguments = parser.parse_args_into_dataclasses()[1]
    
    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.model_name, padding_side="right")
    model = AutoModelForSeq2SeqLM.from_pretrained(datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length).to(datapipeline_args.device)
    
    logging_args.setup_logging(datapipeline_args.model_name)
    
    test_file = Path(datapipeline_args.test_file)
    
    # Load the dataset
    if os.path.isdir(datapipeline_args.test_file):
        logging_args.log_message(f"Loading dataset from local dir")
        test_files = os.listdir(datapipeline_args.test_file)
        test_files_dir = test_file.as_posix()
        test_output_dir = datapipeline_args.output_file
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
    else:
        logging_args.log_message("Loading dataset from single file")
        test_files = [test_file.name]
        test_files_dir = test_file.parent.as_posix()
        test_output_file = datapipeline_args.output_file
        test_output_dir = None
    
    for i, test_file in enumerate(test_files):
        # Initialize the output file path
        if test_output_dir is not None:
            test_output_file = os.path.join(test_output_dir, test_file)
        test_file_path = os.path.join(test_files_dir, test_file)
        logging_args.log_message(f"Loading dataset from {test_file}")

        dataset = load_dataset(
            "parquet",
            data_files={"test": test_file_path},
            split="test",
            num_proc=os.cpu_count(),
        )       
        #logging_args.log_message(f"Tokenizing dataset {test_file}")
        
        # Tokenize the dataset
        #dataset_tokenized = dataset.map(
        #    lambda example, tokenizer=tokenizer: tokenizer(example["text"], padding=True, return_tensors="pt", truncation=True, max_length=2050),
        #    batched=True,
        #    num_proc=os.cpu_count(),
        #)

        # Remove the columns that are not needed
        #dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    
        dataframe = pd.DataFrame(columns=["text", "label", "generated_0", "time_token", "generation_time", "num_tokens", "num_tokens_input"])
        print(f"Processing {len(dataset)} examples")
        # Generate sequences for the entire dataset
        with torch.no_grad():
            for i, element in enumerate(dataset):
                start_time = time.time()
                inputs = tokenizer(element["text"], return_tensors="pt", truncation=True, max_length=2050).to("cuda")

                if inputs.input_ids.shape[1] > 2048:
                    row = [element["text"], element["label"], None, None, None, None, inputs.input_ids.shape[1]]
                    dataframe.loc[i] = row
                    continue

                
               
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=datapipeline_args.max_gen_length,
                    num_beams=datapipeline_args.num_beams,
                    num_return_sequences=datapipeline_args.num_return_sequences,
                    early_stopping=True if datapipeline_args.num_beams != 1 else False,
                    return_dict_in_generate=False,
                    output_attentions=False
                )
                # Decode the generated sequences
                decoded_sequences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
 

                end_time = time.time()


                # Measure the time after generation
                elapsed_time = end_time - start_time
                total_tokens = sum(len(seq) for seq in generated_ids)
                time_per_token = elapsed_time / total_tokens
                row = [element["text"], element["label"], decoded_sequences[0], time_per_token, elapsed_time, total_tokens, inputs.input_ids.shape[1]]
                dataframe.loc[i] = row

                torch.cuda.empty_cache()
                if i % 100 == 0:
                    logging_args.log_message(torch.cuda.memory_summary(abbreviated=True))
                    logging_args.log_message(f"Processed {i}")

        ### Process the labels
        labels_dataframe = dataframe["label"].apply(process_labels, output_columns=["label_node_type", "label_action"])
        dataframe = pd.concat([labels_dataframe, dataframe], axis=1)
        
        for i in range(datapipeline_args.num_return_sequences):
            generated_dataframe = dataframe[f"generated_{i}"].apply(process_labels, output_columns=[f"generated_{i}_node_type", f"generated_{i}_action"])
            dataframe = pd.concat([dataframe, generated_dataframe], axis=1)

        dataframe.to_parquet(test_output_file)
        logging_args.log_message(f"Results saved to {test_output_file}")
    
