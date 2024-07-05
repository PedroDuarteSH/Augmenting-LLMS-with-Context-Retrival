# Purpose: Test the CodeT5 model on the test dataset.
from dataclasses import dataclass, field
import os
from regex import D
from transformers import (
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    AutoTokenizer,
    )
from datasets import load_dataset, concatenate_datasets
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
    
    del averaged_encoder_attentions
    
    # Map attention weights to input tokens
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_mapping = {}
    for i in range(len(input_tokens)):
        if input_tokens[i] == "<pad>":
            continue
        attention_mapping[f"{input_tokens[i]}%{i}"] = averaged_attention_per_token[0][i].item()
        
    del averaged_attention_per_token
    return attention_mapping

if __name__ == "__main__":
    # Initialize the parser
    parser : HfArgumentParser = HfArgumentParser([TestArguments, LoggerArguments])
    
    # Parse the arguments
    datapipeline_args : TestArguments = parser.parse_args_into_dataclasses()[0]
    logging_args : LoggerArguments = parser.parse_args_into_dataclasses()[1]
    if logging_args.log:
        logging_args.setup_logging(datapipeline_args, logging_args)

    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.model_name, padding_side="right")
    model = AutoModelForSeq2SeqLM.from_pretrained(datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length).to(datapipeline_args.device)
    
    
    
    
    test_file = Path(datapipeline_args.test_file)   
    node_type = datapipeline_args.node_type
    
    
    if os.path.isdir(datapipeline_args.test_file):
        logging_args.log_message("Loading dataset from local dir")
        test_files = os.listdir(datapipeline_args.test_file)
        test_output_dir = test_file.as_posix().split('.')[0] + "_attention_nodes.parquet"
        test_files_dir = test_file.as_posix()   
          
    else:
        logging_args.log_message("Loading dataset from single file")
        print("Loading dataset from single file")
        test_files = [test_file.name]
        test_output_dir = test_file.parent.as_posix().split('.')[0] + "_attention_nodes.parquet"
        test_files_dir = test_file.parent.as_posix()
       
    
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    

    index_file = 0
    test_file = test_files[index_file]
    test_output_file = os.path.join(test_output_dir, node_type)
    test_file_path = os.path.join(test_files_dir, test_file) 
    logging_args.log_message(f"Loading dataset from {test_file}")
    dataset = load_dataset(
        "parquet",
        data_files={"test": test_file_path},
        split="test",
        num_proc=os.cpu_count(),
    )       
    
    logging_args.log_message(f"Tokenizing dataset {test_file}")
    
    print(dataset.column_names)
    # Filter the dataset to only include the node type
    dataset = dataset.filter(lambda example: example['label'].startswith(node_type) )

    while len(dataset) < 100:
        print(len(dataset))
        next_file = index_file + 1
        test_file = test_files[index_file]
        test_file_path = os.path.join(test_files_dir, test_file)
        
        if next_file < len(test_files):
            new_dataset = load_dataset(
                "parquet",
                data_files={"test": test_file_path},
                split="test",
                num_proc=os.cpu_count(),
            )
            new_dataset = new_dataset.filter(lambda example: example['label'].startswith(node_type))
            dataset = concatenate_datasets([dataset, new_dataset])
        else:
            print(f"No more files to load. {len(dataset)} examples loaded")
            break
            
    if len(dataset) == 0:
        print(f"No examples for {node_type} in {test_file}")
        exit()
    
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
    
    # Tokenize the dataset
    dataset_tokenized = dataset.map(
        lambda example, tokenizer=tokenizer: tokenizer(example["text"], padding=True, return_tensors="pt", truncation=True, max_length=2048),
        batched=True,
        num_proc=os.cpu_count(),
    )

    # Remove the columns that are not needed
    dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    dataframe = pd.DataFrame(columns=["label", "generated_0", "generated_1", "generated_2", "generated_3", "generated_4", "time_token", "generation_time", "num_tokens", "num_tokens_input", "encoder_attentions"])
    print(f"Processing {len(dataset_tokenized)} examples")
    # Generate sequences for the entire dataset
    with torch.no_grad():
        for i, element in enumerate(dataset_tokenized):
            # Measure the time before generation
            ids = element["input_ids"].unsqueeze(0).detach().to("cuda")
            mask = element["attention_mask"].unsqueeze(0).detach().to("cuda")
            start_time = time.time()
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=datapipeline_args.max_gen_length,
                num_beams=datapipeline_args.num_beams,
                num_return_sequences=datapipeline_args.num_return_sequences,
                early_stopping=True,
                return_dict_in_generate=True,
                output_attentions=True
                
            )
             
            
            # Decode the generated sequences
            decoded_sequences = tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
            
            end_time = time.time()
            # Extract encoder attentions
            encoder_attentions_token_mapping = encoder_attentions_token(generated_ids.encoder_attentions, element["input_ids"], tokenizer)


            # Measure the time after generation
            elapsed_time = end_time - start_time
            total_tokens = sum(len(seq) for seq in generated_ids.sequences)
            time_per_token = elapsed_time / total_tokens
            row = [element["label"], decoded_sequences[0], decoded_sequences[1], decoded_sequences[2], decoded_sequences[3], decoded_sequences[4], time_per_token, elapsed_time, total_tokens, dataset_tokenized['attention_mask'][i].sum().item(), json.dumps(encoder_attentions_token_mapping)]
            dataframe.loc[i] = row
            del generated_ids
            del mask
            del ids
            torch.cuda.empty_cache()
    

    ### Process the labels
    labels_dataframe = dataframe["label"].apply(process_labels, output_columns=["label_node_type", "label_action"])
    dataframe = pd.concat([labels_dataframe, dataframe], axis=1)
    
    for i in range(datapipeline_args.num_return_sequences):
        generated_dataframe = dataframe[f"generated_{i}"].apply(process_labels, output_columns=[f"generated_{i}_node_type", f"generated_{i}_action"])
        dataframe = pd.concat([dataframe, generated_dataframe], axis=1)

    dataframe.to_parquet(test_output_file)
    logging_args.log_message(f"Results saved to {test_output_file}")
    print(f"Results saved to {test_output_file}")

