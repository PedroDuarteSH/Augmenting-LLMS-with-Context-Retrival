

import pandas as pd
import os
import re
    
if __name__ == "__main__":
    if not os.path.exists("partial_test_python_v1_dataset_results.parquet"):
        path = "msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_results.parquet"
    
        dir = os.listdir(path)
        dataset = pd.read_parquet(os.path.join(path, dir[0]))
        full_dataset = pd.DataFrame(columns=dataset.columns)
        for file in dir:
            partial_dataset = pd.read_parquet(os.path.join(path, file))
            partial_dataset["file"] = file
            full_dataset = pd.concat([full_dataset, partial_dataset])

        full_dataset = full_dataset[full_dataset["num_tokens"] < 2048]
        
        node_types = pd.read_parquet("node_types.parquet")
        partial_dataset = pd.DataFrame()
        for node_type in node_types["node_type"]:
            node_dataset = full_dataset[full_dataset["label_node_type"] == node_type]
            if len(node_dataset) >= 1000:
                node_dataset = node_dataset.sample(1000, random_state=42)

            partial_dataset = pd.concat([partial_dataset, node_dataset[["file", "label_node_type"]]])
        
        partial_dataset.to_parquet("partial_test_python_v1_dataset_results.parquet")
        print(partial_dataset)
    else:
        partial_dataset = pd.read_parquet("partial_test_python_v1_dataset_results.parquet")
        print(partial_dataset)
        print(len(partial_dataset))
        
        path = "msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
        full_dataset = pd.DataFrame()
        for file in partial_dataset["file"].unique():
            file_data = partial_dataset[partial_dataset["file"] == file]
            dataset = pd.read_parquet(os.path.join(path, file))
            dataset = dataset.iloc[file_data.index]
            for i, row in dataset.iterrows():
                if file_data.loc[i, "label_node_type"] not in row["label"]:
                    print(f"ERROR: {file_data.iloc[i]['label_node_type']} not in {row['label']}")
            
            full_dataset = pd.concat([full_dataset, dataset])
            print(file)
        print(len(full_dataset))
        full_dataset.to_parquet("partial_test_python_v1_dataset.parquet")
    