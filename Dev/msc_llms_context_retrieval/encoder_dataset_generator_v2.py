
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
import sys
sys.setrecursionlimit(10000)
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


def get_sequence(key, corresponding_action, sequence, visited, recursion = 1):
    if visited[key]:
        return sequence[key]
    visited[key] = True
    string_sequence = f"{sequence[key]}"
    if key in corresponding_action:
        values = corresponding_action[key]
        for item in values:
            string_sequence += f".{item[1]}({get_sequence(item[0], corresponding_action, sequence, visited, recursion + 1)})"
    return string_sequence




def get_flow_sequence(code):
    """Get the flow sequence for the given code

    Args:
        code (str): The code to get the flow sequence from

    Returns:
        list[str]: The flow sequence
    """

    list_lines = code.split("\n")[:-1]
    sequence = []
    text_sequence = ""

    corresponding_action = {}
    
    action_regex = re.compile(r"\.(\w+)")
    int_regex = re.compile(r"\d+")
    

    for i in range(len(list_lines)):
        if list_lines[i] == "":
            continue
        ipos = list_lines[i].find("I")
        parenthesis = list_lines[i].find("(")

        if ipos != -1 and parenthesis != -1:
            sequence.append(list_lines[i][ipos:parenthesis])
        elif "<End>" in list_lines[i]:    
            sequence.append("<END>")
        


        elif "<extra_id_0>" in list_lines[i]:
            sequence.append("<mask>")
        else:
            action_match = action_regex.findall(list_lines[i])
            int_match = int_regex.findall(list_lines[i])
            if action_match:
                if len(int_match) == 2:
                    int_match_0 = int(int_match[0]) - 1
                    int_match_1 = int(int_match[1]) - 1
                    if not int_match_0 in corresponding_action:
                        corresponding_action[int_match_0] = [(int_match_1, action_match[0])]
                    else:
                        corresponding_action[int_match_0] += [(int_match_1, action_match[0])  ]                 

    visited = [False] * len(sequence)
    key = 0
    text_sequence = get_sequence(key, corresponding_action, sequence, visited)

    return text_sequence


if __name__ == "__main__":
    """Script to process the dataset, to the encoder form and prepare it for the training process"""
    parser = HfArgumentParser([EncoderDatasetGeneratorArguments])
    datapipeline_args: EncoderDatasetGeneratorArguments = parser.parse_args_into_dataclasses()[0]
    

    if not os.path.isdir(datapipeline_args.full_dataset_path):
        output_path = datapipeline_args.output_dataset_path
        flow_only_files = [datapipeline_args.flow_only_dataset_path]
        flow_only_dir = Path(datapipeline_args.flow_only_dataset_path).parent.as_posix()

    else:
        output_path = None
        output_dir_path = datapipeline_args.output_dataset_path  
        os.makedirs(output_dir_path, exist_ok=True)
        flow_only_files = os.listdir(datapipeline_args.flow_only_dataset_path)
        flow_only_dir = datapipeline_args.flow_only_dataset_path
    
    
    
    # Initialize tokenizer and model
    for file in flow_only_files:
        file = file.split("/")[-1]
        print(f"Processing {file}")
        flow_only_path = os.path.join(flow_only_dir, file)
        if output_path is None:
            output_path = os.path.join(output_dir_path, file)
        if os.path.exists(output_path):
            print("File Already processed")
            output_path = None
            continue
        
        dataframe = pd.read_parquet(flow_only_path)

        
        dataframe["label"] = dataframe["label"].apply(utils.process_labels, output_columns=["node_type", "action"])["node_type"]

        dataframe["text"] = dataframe.apply(lambda x: get_flow_sequence(x["text"]), axis=1)
        dataframe.to_parquet(output_path)
        output_path = None
   


## 