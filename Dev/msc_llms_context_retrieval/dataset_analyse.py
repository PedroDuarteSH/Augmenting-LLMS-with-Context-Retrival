import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import utils
if __name__ == "__main__":
    path = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only_encoded.parquet"
    path_test="msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_flow_only_encoded_v2.parquet"
    dataset = load_dataset(path, split="train")
    dataset_test = load_dataset(path_test, split="train")
    print(dataset_test)
    dataset = concatenate_datasets([dataset, dataset_test])
    print(dataset)

    dataset = dataset.select_columns("node_type_int")
    dataset = dataset.to_pandas()
    print(dataset["node_type_int"].value_counts())
    dataset_node_types = dataset['node_type_int'].value_counts().reset_index()
    dataset_node_types.columns = ['node_type_int', 'count']
    
    # Map the node types to their names
    dataset_node_types['node_type_name'] = dataset_node_types['node_type_int'].map(lambda x: utils.id2nodetype()[x])
    
    # Create the histogram
    fig = px.bar(dataset_node_types, x='node_type_name', y='count', text="count", title="Node type distribution in the entire dataset", labels={"node_type_name": "Node type", "count": "Frequency"})
    #fig.add_trace(go.Histogram(dataset_node_types, x='count', name="count", texttemplate="%{x}", textfont_size=20))
    fig.show()
    print(dataset)
    print(len(dataset))
