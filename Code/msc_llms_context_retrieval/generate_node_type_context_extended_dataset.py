   
from datasets import load_dataset
from CodeSplitter import CodeSplitter
import re
import pandas as pd
import utils
import os

from transformers import HfArgumentParser
from params_v2 import ProcessDataset
from pathlib import Path
cd = CodeSplitter()
pattern_extra_id = r"var\d+ = <extra_id_\d+>"
pattern_extra_id_regex = re.compile(pattern_extra_id)

def get_label_relevant_words(label):
    regex = re.compile(r"\b\w+\b(?=[\s{},.():\]])")

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
    relevant_words = set(relevant_words)
    return relevant_words


def add_entity_to_text(text : str, entities : list[str]) -> str:
    """Add the entities to the flow text

    Args:
        text (str): Input text
        entities (list[str]): List of entities to add to the text

    Returns:
        str: The text with the entities added
    """
    
   
    flow = cd.get_flow(text)
    if entities is None:
        return flow
    
    extra_id_matches = pattern_extra_id_regex.search(flow).span() if pattern_extra_id_regex.search(flow) is not None else None

    comment = f"# Entities: {entities}\n"
    flow = flow[:extra_id_matches[0]] + comment + flow[extra_id_matches[0]:]
    
    return flow

## Label is a integer that represents the context that should be used for the model
# 0. "full_context"
# 1. "flow_only"
# 2. "flow_and_imports"
# 3. "flow_and_dataclasses"
def get_context_label(text : str, relevant_words : list[str], label : str) -> int:
    """Get the context label for the given text and relevant words

    Args:
        text (str): Input text for the context selection
        relevant_words (list[str]): List of relevant words for the context selection

    Returns:
        int: The context label for the given text
    """

    # Get the code sections separated
    imports, flow, dataclasses = cd.split_code(text)
    dataclasses_bool = False
    dataclasses_list = []
    relevant_words_flow = get_label_relevant_words(flow)
    for dataclass in dataclasses:
        dataclasse_name = utils.get_dataclass_name(dataclass)
        if dataclasse_name in relevant_words_flow:
            dataclasses_list.append(dataclasse_name)

        

    # Check if the relevant words are in the imports or dataclasses
    imports_bool = False
    
    for word in relevant_words:
        regex = re.compile(rf"(?<=[ ,=({{]){word}\b")
        # Check if the word is in the imports or dataclasses
        if not imports_bool and regex.search(imports):
            imports_bool = True
        for class_text in dataclasses:
            #print(class_text)
            if word in class_text:
                class_name = utils.get_dataclass_name(class_text)
                dataclasses_list.append(class_name)
                dataclasses_bool = True


    # Join the dataclasses list
    dataclasses_list = ", ".join(set(dataclasses_list))
   

    # If the word is in both imports and dataclasses, return the full context
    if imports_bool and dataclasses_bool:
        return "Full: " + dataclasses_list
    # If the word is in the imports, return the flow and imports context
    elif imports_bool:
        return "Imports"
    # If the word is in the dataclasses, return the flow and dataclasses context
    elif dataclasses_bool:
        return "Dataclasses: " + dataclasses_list
    # If the word is not in any of the extra contexts, return the flow only context
    return "Flow"

if __name__ == "__main__":  
    parser = HfArgumentParser([ProcessDataset])
    datapipeline_args: ProcessDataset = parser.parse_args_into_dataclasses()[0]
    

    print(datapipeline_args)

    d_path = Path(datapipeline_args.dataset_path)
    if not os.path.isdir(datapipeline_args.dataset_path):
        output_path = datapipeline_args.output_path
        dataset_files = [d_path.name]
        dataset_dir = d_path.parent.as_posix()


    else:
        output_path = None
        output_dir_path = datapipeline_args.output_path  
        os.makedirs(output_dir_path, exist_ok=True)    
        dataset_dir = datapipeline_args.dataset_path
        dataset_files = os.listdir(datapipeline_args.dataset_path)


    
    # Initialize tokenizer and model
    for file in dataset_files:
        print(f"Processing {file}")
        full_context_path = os.path.join(dataset_dir, file)
        
        if output_path is None:
            output_path = os.path.join(output_dir_path, file)
        
        dataset = pd.read_parquet(full_context_path)
        dataset["relevant_words"] = dataset["label"].apply(get_label_relevant_words)
        dataset["context"] = dataset.apply(lambda x: get_context_label(x["text"], x["relevant_words"], x["label"]), axis=1)
        dataset[['node_type', 'action']] = dataset['label'].apply(utils.process_labels)
        dataset["label"] = dataset.apply(lambda x: utils.node_type_context_to_label(x["node_type"], x["context"]), axis=1)

        dataset["entities"] = dataset["text"].apply(utils.get_dataclass_names)
        dataset["text"] = dataset.apply(lambda x: add_entity_to_text(x["text"], x["entities"]), axis=1)

        dataset.to_parquet(output_path)
        output_path = None