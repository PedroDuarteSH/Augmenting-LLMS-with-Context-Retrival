import pandas as pd
import utils
import os
from pathlib import Path
import json
import time
import re

def update_dictionary(dictionary, node_type, left_action):
    if left_action is None:
        return

    if node_type not in dictionary:
        dictionary[node_type] = set(action[0] for action in left_action)
    else:
        dictionary[node_type].update(action[0] for action in left_action)
        
def get_dictionary_of_properties(file_list, is_dir = False, path = None):
    if os.path.exists("node_properties.json"):
        with open("node_properties.json", "r") as infile:
            dictionary = json.load(infile)
        return dictionary
   
    for file in file_list:
        print(f"Processing {file}")
        ## Find node parameters in labels
        start = time.time()
        dataset = pd.read_parquet(file if not is_dir else os.path.join(path, file))
        end = time.time()
        print(f"Reading parquet file took {end-start} seconds")
        start = time.time()
        labels = dataset["label"].apply(utils.process_labels, output_columns = ["node_type", "left_action"])
        end = time.time()
        print(f"Processing labels took {end-start} seconds")
        labels.reset_index(drop=True, inplace=True)
        start = time.time()
        labels = labels.apply(lambda x: update_dictionary(dictionary, x["node_type"], x["left_action"]), axis=1)
        end = time.time()
        print(f"Updating dictionary took {end-start} seconds")
    for key in dictionary:
        dictionary[key] = list(dictionary[key])
    json_object = json.dumps(dictionary, indent = 4) 
    with open("node_properties.json", "w") as outfile:
        outfile.write(json_object)
    return dictionary 

def replace_text(row, dictionary):
    left, right = row["extra_id_location"]
    command :str = row["text"][left:right+1]
    
    if row["node_type"] not in dictionary:
        return row
    
    if len(dictionary[row["node_type"]]) > 0:
        comment = f"# Properties: {', '.join(dictionary[row['node_type']])}\n"
    else:
        comment = f"# Properties: \n"
    command = command.replace("<extra_id_0>", f"{row['node_type']}(<extra_id_0>)")
   
    row["text"] = row["text"][:left] + comment + command + row["text"][right+1:]
    row["label"] = ', '.join([f"{action[0]}={action[1]}" for action in row["left_action"]])
   
    return row

if __name__ == "__main__":
    print("Hello, World!")
    path = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset.parquet"


    file_path = Path(path)
    if file_path.is_dir():
        file_list = os.listdir(path)
        output_path = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_properties.parquet/"
        os.makedirs(output_path, exist_ok=True)
        is_dir = True
    else:
        file_list = [path]
        is_dir = False
        output_path = path.split(".")[0] + "_properties.parquet"
    dictionary = get_dictionary_of_properties(file_list, is_dir, path)
    print(dictionary)

    for i, file in enumerate(file_list):
        print(f"Processing {file} ({i})")
        dataset = pd.read_parquet(file if not is_dir else os.path.join(path, file))
        pattern_extra_id = r"var\d+ = <extra_id_\d+>"
        pattern_extra_id_regex = re.compile(pattern_extra_id)

        dataset["extra_id_location"] = dataset["text"].apply(lambda x: pattern_extra_id_regex.search(x).span() if pattern_extra_id_regex.search(x) is not None else None)
        dataset[["node_type", "left_action"]] = dataset["label"].apply(utils.process_labels, output_columns = ["node_type", "left_action"])
        #print(dataset)
        
        dataset = dataset.apply(lambda row: replace_text(row, dictionary), axis=1)
        #print(a)
        dataset.drop(columns=["extra_id_location"], inplace=True)

        dataset.to_parquet(output_path + file if is_dir else output_path, index=False)
        

