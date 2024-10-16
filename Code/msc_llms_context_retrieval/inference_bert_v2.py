# Purpose: Test the CodeT5 model on the test dataset.
from dataclasses import dataclass, field
import os
from regex import D
from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    AutoTokenizer,
    )
from datasets import load_dataset
import pandas as pd
import time
from pathlib import Path

import torch
from params_v2 import TestArguments, LoggerArguments
from utils import process_labels
import utils



if __name__ == "__main__":
    # Initialize the parser
    parser : HfArgumentParser = HfArgumentParser([TestArguments, LoggerArguments])
    
    # Parse the arguments
    datapipeline_args : TestArguments = parser.parse_args_into_dataclasses()[0]
    logging_args : LoggerArguments = parser.parse_args_into_dataclasses()[1]
    
    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.model_name, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length, trust_remote_code=True).to(datapipeline_args.device)
    
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
        logging_args.log_message(f"Tokenizing dataset {test_file}")
        

        dataframe = pd.DataFrame(columns=["label", "label_node_type", "output_node_type", "inference_time", "num_tokens_input"])
        print(f"Processing {len(dataset)} examples")

        # Generate sequences for the entire dataset
        with torch.no_grad():
            for i, element in enumerate(dataset):
                # Measure the time before generation
                start_time = time.time()
                
                x = tokenizer(element["text"], padding=True, return_tensors="pt").to(datapipeline_args.device)
                element["input_ids"], element["attention_mask"] = x["input_ids"], x["attention_mask"]
      
                if element["attention_mask"].sum().item() < 2048:
                    row = [element["label"], element["node_type_int"], None, None, element["attention_mask"].sum().item()]
                    dataframe.loc[i] = row
                    continue
                
                outputs = model(**x)

                node_type = outputs.logits[0][:].argmax().item()
                print(node_type)
                #context_type = outputs.logits[0][utils.context_indices()].argmax().item()

                
                end_time = time.time()
                elapsed_time = end_time - start_time

                row = [element["label"], element["node_type_int"], node_type, elapsed_time, element["attention_mask"].sum().item()]
                dataframe.loc[i] = row

        dataframe.to_parquet(test_output_file)
        logging_args.log_message(f"Results saved to {test_output_file}")
    
