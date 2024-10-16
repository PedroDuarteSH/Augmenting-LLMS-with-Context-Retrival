
import os
from platform import node
import re
import pandas as pd
from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field
import numpy as np
import utils

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
    
#
# Resposta depende do que está 
# Deep Seek 
# Estruturar e documentar o que já correu
# 


@dataclass
class MetricsArguments:
    generated_processed: str = field(
        metadata={"help": "Path to the generated path file."},
        default="msc_llms_context_retrieval/datasets/logic/node(context)_2step_model_end2end_test.parquet",
        #default="msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_all_properties_results.parquet"
    )
    num_generated_sequences: int = field(
        metadata={"help": "Number of sequences to generate."},
        default=5
    )

def left_node_action_accuracy(row, num_generated_sequences):

  

    for i in range(num_generated_sequences):
        ## If the generated action is <END>, then it is correct
        if row["label_node_type"] == "<END>" and row[f"node_type"] == "<END>":
            return 1
        
        
        if row["label_node_type"] != row["node_type"]:
            return 0

        
        if row["label_action"] is None and row[f"generated_{i}_action"] is None:
            return 1

        if row["label_action"] is None:
            continue
    
        # Check if generated node type is None
        if row[f"generated_{i}_action"] is None:
            continue
        l = len(row["label_action"])

        


        
        for label_left_action in row["label_action"]:
            for k in range(len(row[f"generated_{i}_action"])):
                generated_left_var = row[f"generated_{i}_action"][k]
                if label_left_action[0] == generated_left_var[0] and label_left_action[1].strip("[]") == generated_left_var[1].strip("[]"):
                    l -= 1
                    break
        if l == 0:
           
            return 1
    return 0
    

if __name__ == "__main__":
    
    # Initialize the parser with MetricsArguments
    parser : HfArgumentParser = HfArgumentParser([MetricsArguments])
    # Parse the arguments
    datapipeline_args : MetricsArguments = parser.parse_args_into_dataclasses()[0]
    results =  pd.DataFrame()
    for file in os.listdir(datapipeline_args.generated_processed):
        print(file)
        results_temp = pd.read_parquet(os.path.join(datapipeline_args.generated_processed, file))
        
        results = pd.concat([results, results_temp])
    #results = pd.read_parquet(datapipeline_args.generated_processed)
    print(results.columns)
    print(len(results))

    #filter_num_tokens = pd.read_csv("test.csv")

    #print(filter_num_tokens["num_tokens_input"] == True)
    #print((filter_num_tokens["num_tokens_input"] == True).sum())
    results.reset_index(drop=True, inplace=True)
    
    for i in range(datapipeline_args.num_generated_sequences):
        results[f"generated_{i}_action"] = results[f"generated_{i}"].apply(utils.match_action)
    
    dataframe = pd.DataFrame(columns=["Node Type", "Node Type Accuracy", "Full Matching", "Full Matching with Right Node Type", "Num Samples"])
    
    
    node_type_accuracy = (results["label_node_type"] == results["node_type"]).sum() / len(results)
  
    results["property_accuracy"] = results.apply(lambda x: left_node_action_accuracy(x, datapipeline_args.num_generated_sequences), axis=1)
    
    newdf = pd.DataFrame({
        "nodetypeacc": results["label_node_type"] == results["node_type"],
        "left_action_acc": results["property_accuracy"]
    })
    newdf.to_csv("./metrics/2_step_details.csv")
    property_accuracy = results["property_accuracy"].sum() / len(results)
    #node_type_properties_results = results[results["label_node_type"] == results["node_type"]]

    #property_right_accuracy = node_type_properties_results["property_accuracy"].sum() / len(node_type_properties_results)
  
    property_right_accuracy = property_accuracy/node_type_accuracy
    dataframe.loc[0] = ["Global", node_type_accuracy, property_accuracy, property_right_accuracy, len(results)]
    for node_type in node_type_list:
        #print(f"\tNode type: {node_type}")
        node_type_results = results[results["label_node_type"] == node_type]
        #node_type_properties_results = node_type_results[node_type_results["label_node_type"] == node_type_results["node_type"]]
        



        if len(node_type_results) == 0:
            #print("\t\tNo instances for this node type\n")
            continue
        node_type_accuracy = (node_type_results["label_node_type"] == node_type_results["node_type"]).sum() / len(node_type_results)
   
        #print(f"\t\tNode type accuracy: {node_type_accuracy}")
        property_accuracy = node_type_results["property_accuracy"].sum() / len(node_type_results)
        #print(f"\t\tProperty accuracy: {property_accuracy}")
        
        #property_right_accuracy = node_type_properties_results["property_accuracy"].sum() / len(node_type_properties_results)

        property_right_accuracy = property_accuracy/node_type_accuracy

        #print(f"\t\tProperty right accuracy: {property_right_accuracy}")
        print(f"Node type: {node_type}")
        print(node_type_results["context"].value_counts())
        print(node_type_results["num_tokens"].describe())
        #print("\n")
        dataframe.loc[len(dataframe)] = [node_type, node_type_accuracy, property_accuracy, property_right_accuracy, len(node_type_results)]

    print(dataframe)
    fileName = datapipeline_args.generated_processed.split("/")[-1]
    fileName = fileName.split(".")[0]
    dataframe.to_csv(f"metrics/{fileName}_metrics.csv", index=False)
