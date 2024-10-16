
import os
from platform import node
import re
import pandas as pd
from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field
import numpy as np


node_type_list = ["IIfNode",
    "IJSONSerializeNode",
    "INRSendEmailNode",
    "IAssignment",
    "ISendEmailNode",
    "IRaiseExceptionNode",
    "IRecordListToExcelNode",
    "IForEachNode",
    "IExcelToRecordListNode",
    "ISQLNode",
    "IAggregateNode",
    "<END>",
    "IExecuteServerActionNode",
    "IJSONDeserializeNode",
]
    

@dataclass
class MetricsArguments:
    generated_processed: str = field(
        metadata={"help": "Path to the generated path file."},
        default="msc_llms_context_retrieval/datasets/logic/node(context)_model_step1_results.parquet"
    )
    num_generated_sequences: int = field(
        metadata={"help": "Number of sequences to generate."},
        default=5
    )


def accuracy(list1, list2):
    """Get the accuracy of the list1 compared to list2"""
    return sum([1 if list1[i] == list2[i] else 0 for i in range(len(list1))]) / len(list1)


def process_node_context(labels, column_names=["node_type", "context"]):
    """Process the labels to return the node type and action"""
    # Split the labels into node types and actions
    split_labels = labels.str.split("(", expand=True)
    node_types = split_labels[0]
    actions = split_labels[1].str.rstrip(")")

    # Create a DataFrame with the node types and actions
    processed_labels = pd.DataFrame({
        column_names[0]: node_types,
        column_names[1]: actions
    })


    return processed_labels

def context_accuracy_including_more(prevision : str, true : str) -> int:
    if prevision == "Full":
        return 1
    if (prevision == "Dataclasses" or prevision == "Imports") and true == "Flow":
        return 1
    return prevision == true
    
    
    

if __name__ == "__main__":
    
    # Initialize the parser with MetricsArguments
    parser : HfArgumentParser = HfArgumentParser([MetricsArguments])
    # Parse the arguments
    datapipeline_args : MetricsArguments = parser.parse_args_into_dataclasses()[0]
    results = pd.read_parquet(datapipeline_args.generated_processed)


    #filter_num_tokens = pd.read_csv("test.csv")

    #print(filter_num_tokens["num_tokens_input"] == True)
    #print((filter_num_tokens["num_tokens_input"] == True).sum())
    results.reset_index(drop=True, inplace=True)

    #results = results[filter_num_tokens["num_tokens_input"] == True]
    
    # Apply the function to the 'label' and 'generated_0' columns
    results[["label_node_type", "context_label"]] = results["label"].str.split("(", expand=True, n=1)
    results["context_label"] = results["context_label"].str.rstrip(")")
    
    dataframe = pd.DataFrame(columns=["Node Type", "Node Type Accuracy", "Context Accuracy", "Context accuracy including more than predicted", "Num Samples"])

    results[["generated_0_node_type", "generated_0_context"]] = results["generated_0"].str.split("(", expand=True)
    results["generated_0_context"] = results["generated_0_context"].str.rstrip(")")

    
    context_accuracy_more = results.apply(lambda x: context_accuracy_including_more(x["generated_0_context"], x["context_label"]), axis=1).sum() / len(results)

    
    context_accuracy = (results["generated_0_context"] == results["context_label"]).sum() / len(results)
    
    node_type_accuracy = (results["label_node_type"] == results["generated_0_node_type"]).sum() / len(results)
    

    
    dataframe.loc[len(dataframe)] = ["Global", node_type_accuracy, context_accuracy, context_accuracy_more, len(results)]

    for node_type in node_type_list:
        
        node_type_results = results[results["label_node_type"] == node_type]
        if len(node_type_results) == 0:
           
            continue
        context_accuracy = (node_type_results["generated_0_context"] == node_type_results["context_label"]).sum() / len(node_type_results)
        context_accuracy_more = node_type_results.apply(lambda x: context_accuracy_including_more(x["generated_0_context"], x["context_label"]), axis=1).sum() / len(node_type_results)
        node_type_accuracy = (node_type_results["label_node_type"] == node_type_results["generated_0_node_type"]).sum() / len(node_type_results)
        
        dataframe.loc[len(dataframe)] = [node_type, node_type_accuracy, context_accuracy, context_accuracy_more, len(node_type_results)]

    print(dataframe)

    fileName = datapipeline_args.generated_processed.split("/")[-1]
    fileName = fileName.split(".")[0]
    dataframe.to_csv(f"metrics/{fileName}_metrics.csv", index=False)

