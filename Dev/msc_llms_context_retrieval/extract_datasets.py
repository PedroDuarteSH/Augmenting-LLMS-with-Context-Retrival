from datasets import load_dataset
import os
import re
import pandas as pd
from pathlib import Path
from params_v2 import ExtractDatasetArguments
from transformers import HfArgumentParser
from CodeSplitter import CodeSplitter



if __name__ == "__main__":
    # Initialize the parser
    parser : HfArgumentParser = HfArgumentParser([ExtractDatasetArguments])
    
    # Parse the arguments
    datapipeline_args : ExtractDatasetArguments = parser.parse_args_into_dataclasses()[0]

    dataset_path = Path(datapipeline_args.dataset_path)
    output_path = Path(datapipeline_args.output_path)

    if dataset_path.is_dir():
        file_list = os.listdir(dataset_path)
        dataset_dir = dataset_path.as_posix()
        output_dir = output_path.as_posix()

        output_dir_flow_imports = os.path.join(output_dir.split(".")[0] + '_flow_imports.parquet') 
        output_dir_flow_only = os.path.join(output_dir.split(".")[0] + '_flow_only.parquet')
        output_dir_classes_flows = os.path.join(output_dir.split(".")[0] + '_classes_flows.parquet')

        os.makedirs(output_dir_flow_imports, exist_ok=True)
        os.makedirs(output_dir_flow_only, exist_ok=True)
        os.makedirs(output_dir_classes_flows, exist_ok=True)
    else:
        dataset_dir = dataset_path.parent.as_posix()
        output_dir = None
        output_file = output_path.as_posix()
        file_list = [dataset_path.parts[-1]]

    code_splitter = CodeSplitter()
    for file in file_list:
        dataset = load_dataset(
            "parquet",
            data_files=dataset_dir + "/" + file,
            split="train",
            num_proc=os.cpu_count(),
        )
        dataset = dataset.to_pandas()
        dataset_flow_imports = dataset.copy()
        dataset_flow_imports["text"] = dataset_flow_imports["text"].apply(lambda x: code_splitter.get_imports_and_flow(x))
        dataset_flow_only = dataset.copy()
        dataset_flow_only["text"] = dataset_flow_only["text"].apply(lambda x: code_splitter.get_flow(x))
        dataset_classes_flows = dataset.copy()
        dataset_classes_flows["text"] = dataset_classes_flows["text"].apply(lambda x: code_splitter.get_dataclasses_and_flow(x))
        if output_dir is not None:
            output_file_flow_imports = os.path.join(output_dir_flow_imports, file)
            output_file_flow_only = os.path.join(output_dir_flow_only, file)
            output_file_classes_flows = os.path.join(output_dir_classes_flows, file)
           
        else:
            output_file_flow_imports = output_file.split(".")[0] + '_flow_imports.parquet'
            output_file_flow_only = output_file.split(".")[0] + '_flow_only.parquet'
            output_file_classes_flows = output_file.split(".")[0] + '_classes_flows.parquet'
        
        dataset_flow_imports.to_parquet(output_file_flow_imports)
        dataset_flow_only.to_parquet(output_file_flow_only)
        dataset_classes_flows.to_parquet(output_file_classes_flows)
        
        print(f"Processed {file}")
    