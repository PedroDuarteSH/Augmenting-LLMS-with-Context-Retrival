
from datasets import load_dataset
from CodeSplitter import CodeSplitter
import re
import pandas as pd
import utils
import os

from transformers import HfArgumentParser
from params_v2 import EncoderDatasetGeneratorArguments
from pathlib import Path
cd = CodeSplitter()
def get_label_relevant_words(label):
    regex = re.compile(r"\b\w+\b(?=[\s{},.()\]])")

    relevant_words_iter = regex.finditer(label)
    relevant_words = set()
    last_end = -5
    for match in relevant_words_iter:
        if match.start() == 0:
            last_end = match.end()
            continue

        if last_end+1 == match.start():
            last_end = match.end()
            continue
        last_end = match.end()
        relevant_words.add(match.group(0))
    relevant_words = list(relevant_words)
    return relevant_words


## Label is a integer that represents the context that should be used for the model
# 0. "full_context"
# 1. "flow_only"
# 2. "flow_and_imports"
# 3. "flow_and_dataclasses"
def get_context_label(text : str, relevant_words : list[str]) -> int:
    """Get the context label for the given text and relevant words

    Args:
        text (str): Input text for the context selection
        relevant_words (list[str]): List of relevant words for the context selection

    Returns:
        int: The context label for the given text
    """

    # Get the code sections separated
    imports, _, dataclasses = cd.split_code(text)

    # Check if the relevant words are in the imports or dataclasses
    imports_bool = False
    dataclasses_bool = False

    for word in relevant_words:
        regex = re.compile(rf"(?<=[ ,=({{]){word}\b")
        # Check if the word is in the imports or dataclasses
        if not imports_bool and regex.search(imports):
            imports_bool = True
        for class_text in dataclasses:
            if not dataclasses_bool and regex.search(class_text):
                dataclasses_bool = True
        # If the word is in both imports and dataclasses, return the full context
        if imports_bool and dataclasses_bool:
            return 0
    # If the word is in the imports, return the flow and imports context
    if imports_bool and not dataclasses_bool:
        return 2
    # If the word is in the dataclasses, return the flow and dataclasses context
    if not imports_bool and dataclasses_bool:
        return 3
    # If the word is not in any of the extra contexts, return the flow only context
    return 1

if __name__ == "__main__":
    """Script to process the dataset, to the encoder form and prepare it for the training process"""
    parser = HfArgumentParser([EncoderDatasetGeneratorArguments])
    datapipeline_args: EncoderDatasetGeneratorArguments = parser.parse_args_into_dataclasses()[0]
    

    if not os.path.isdir(datapipeline_args.full_dataset_path):
        output_path = datapipeline_args.output_dataset_path
        full_context_files = [datapipeline_args.full_dataset_path]
        full_context_dir = Path(datapipeline_args.full_dataset_path).parent.as_posix()
        flow_only_files = [datapipeline_args.flow_only_dataset_path]
        flow_only_dir = Path(datapipeline_args.flow_only_dataset_path).parent.as_posix()

    else:
        output_path = None
        output_dir_path = datapipeline_args.output_dataset_path  
        os.makedirs(output_dir_path, exist_ok=True)    
        full_context_dir = datapipeline_args.full_dataset_path
        full_context_files = os.listdir(datapipeline_args.full_dataset_path)
        flow_only_files = os.listdir(datapipeline_args.flow_only_dataset_path)
        flow_only_dir = datapipeline_args.flow_only_dataset_path
    assert full_context_files == flow_only_files

    
    # Initialize tokenizer and model
    for file in full_context_files:
        print(f"Processing {file}")
        full_context_path = os.path.join(full_context_dir, file)
        flow_only_path = os.path.join(flow_only_dir, file)
        if output_path is None:
            output_path = os.path.join(output_dir_path, file)
        
        full_dataset = pd.read_parquet(full_context_path)
        flow_only_dataset = pd.read_parquet(flow_only_path)


        dataset_label_node_type = full_dataset["label"].apply(utils.process_labels, output_columns=["node_type", "action"])
        full_dataset["node_type"] = dataset_label_node_type["node_type"]
        del dataset_label_node_type

        full_dataset["node_type_int"] = full_dataset["node_type"].apply(lambda x: utils.label2id()[x])
       
        

        flow_only_dataset["label"] = full_dataset.apply(lambda x: utils.label_to_bin_node_only(x["node_type_int"]), axis=1)
        flow_only_dataset["node_type_int"] = full_dataset["node_type_int"]
        
        flow_only_dataset.to_parquet(output_path)
        output_path = None
   

