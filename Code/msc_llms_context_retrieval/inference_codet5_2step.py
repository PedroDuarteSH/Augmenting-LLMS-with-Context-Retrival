# Purpose: Test the CodeT5 model on the test dataset.
from dataclasses import dataclass, field
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    AutoTokenizer,
    )
from datasets import load_dataset
import pandas as pd
import time
from pathlib import Path
from CodeSplitter import CodeSplitter
import torch
from params_v2 import TestArguments, LoggerArguments
from utils import process_labels
import json
import utils
import re
def get_dictionary_of_properties():
    if os.path.exists("node_properties.json"):
        with open("node_properties.json", "r") as infile:
            dictionary = json.load(infile)
        return dictionary

if __name__ == "__main__":
    # Initialize the parser
    parser : HfArgumentParser = HfArgumentParser([TestArguments, LoggerArguments])
    
    # Parse the arguments
    datapipeline_args : TestArguments = parser.parse_args_into_dataclasses()[0]
    logging_args : LoggerArguments = parser.parse_args_into_dataclasses()[1]
    
    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.model_name, padding_side="right")
    model = AutoModelForSeq2SeqLM.from_pretrained(datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length).to(datapipeline_args.device)
    model_properties = AutoModelForSeq2SeqLM.from_pretrained(datapipeline_args.model_properties_name, max_length=datapipeline_args.max_gen_length).to(datapipeline_args.device)
    logging_args.setup_logging(datapipeline_args.model_name)
    
    test_file = Path(datapipeline_args.test_file)
    code_splitter = CodeSplitter()
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
    dictionary = get_dictionary_of_properties()
    pattern_extra_id = r"var\d+ = <extra_id_\d+>"
    pattern_extra_id_regex = re.compile(pattern_extra_id)

        

    print(dictionary)
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

    
        dataframe = pd.DataFrame(columns=["text", "label", "node_type", "context", "generated_0", "generated_1", "generated_2", "generated_3" ,"generated_4", "time_token", "generation_time", "num_tokens", "num_tokens_input_node_type", "num_tokens_input_properties"])
        print(f"Processing {len(dataset)} examples")
        # Generate sequences for the entire dataset
        with torch.no_grad():
            for i, element in enumerate(dataset):
                start_time = time.time()
            
                flow = code_splitter.get_flow(element["text"])

                inputs = tokenizer(flow, return_tensors="pt", truncation=True, max_length=2050).to("cuda")
                num_tokens_input_node_type = inputs.input_ids.shape[1]
                if num_tokens_input_node_type > 2048:
                    # "text", "label", "node_type", "context", "generated_0", "generated_1", "generated_2", "generated_3" ,"generated_4", "time_token", "generation_time", "num_tokens", "num_tokens_input_node_type", "num_tokens_input_properties"
                    row = [element["text"], element["label"], None, None, None, None, None, None, None, None, None, None, num_tokens_input_node_type, None]
                    dataframe.loc[i] = row
                    continue

                
                ### Generate node type context
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=datapipeline_args.max_gen_length,
                )
                # Decode the generated sequences
                node_type_context = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                node_type, context = node_type_context.split("(")
                context = context[:-1]
                context = "Full"
                
                element["extra_id_location"] = pattern_extra_id_regex.search(element["text"]).span() if pattern_extra_id_regex.search(element["text"]) is not None else None
                
                if node_type not in dictionary:
                    comment = f"# Properties: \n"
                else:
                    comment = f"# Properties: {', '.join(dictionary[node_type])}\n" if len(dictionary[node_type]) > 0 else f"# Properties: \n"
                element["text"] = element["text"][:element["extra_id_location"][0]] +  comment + element["text"][element["extra_id_location"][0]:]
                element["text"] = element["text"].replace("<extra_id_0>", f"{node_type}(<extra_id_0>)")
                
                ## Get Flow
                selected_context = code_splitter.get_flow(element["text"])
                if context == "Dataclasses":
                    selected_context = code_splitter.get_dataclasses_and_flow(element["text"])
                elif context == "Imports":
                    selected_context = code_splitter.get_imports_and_flow(element["text"])
                elif context == "Full":
                    selected_context = element["text"]

                inputs = tokenizer(selected_context, return_tensors="pt", truncation=True, max_length=2050).to("cuda")
                num_tokens_input_properties = inputs.input_ids.shape[1]
                if num_tokens_input_properties > 2048:
                    # "text", "label", "node_type", "context", "generated_0", "generated_1", "generated_2", "generated_3" ,"generated_4", "time_token", "generation_time", "num_tokens", "num_tokens_input_node_type", "num_tokens_input_properties"
                    row = [element["text"], element["label"], node_type, context, None, None,None, None, None, None, None, None, num_tokens_input_node_type, num_tokens_input_properties]
                    dataframe.loc[i] = row
                    continue

                generated_ids = model_properties.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=datapipeline_args.max_gen_length,
                    num_beams=datapipeline_args.num_beams,
                    num_return_sequences=datapipeline_args.num_return_sequences,
                    early_stopping=True if datapipeline_args.num_beams != 1 else False,
                    return_dict_in_generate=False,
                    output_attentions=False
                )
                decoded_sequences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) 


                end_time = time.time()


                # Measure the time after generation
                elapsed_time = end_time - start_time
                total_tokens = sum(len(seq) for seq in generated_ids)
                time_per_token = elapsed_time / total_tokens
                row = [element["text"], element["label"],  node_type, context, decoded_sequences[0], decoded_sequences[1],decoded_sequences[2],decoded_sequences[3],decoded_sequences[4],time_per_token, elapsed_time, total_tokens, num_tokens_input_node_type, num_tokens_input_properties]
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
    
