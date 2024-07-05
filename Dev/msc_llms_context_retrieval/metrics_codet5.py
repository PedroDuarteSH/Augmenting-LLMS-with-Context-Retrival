
import utils
import os
from platform import node
import re
from unittest import result
import pandas as pd
from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field



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
        default="msc_llms_context_retrieval/datasets/logic/baseline_results_other.parquet"
    )
    num_generated_sequences: int = field(
        metadata={"help": "Number of sequences to generate."},
        default=5
    )


def node_accuracy(row, num_generated_sequences):
    for i in range(num_generated_sequences):
        if row["label_node_type"] == row[f"generated_{i}_node_type"]:
            return 1
    return 0

def check_right_node_type(row, num_generated_sequences):
    for i in range(num_generated_sequences):
        if row["label_node_type"] == row[f"generated_{i}_node_type"]:
            return 1
    return 0

def left_node_action_accuracy(row, num_generated_sequences):
    for i in range(num_generated_sequences):
        ## If the generated action is <END>, then it is correct
        if row["label_node_type"] != row[f"generated_{i}_node_type"]:
            continue
        
        if row["label_action"] is None and row[f"generated_{i}_action"] is None:
            return 1       

        if row["label_action"] is None or row[f"generated_{i}_action"] is None:
            continue
        

        l = len(row["label_action"])

        for label_left_action in row["label_action"]:
            #print(label_left_action)
            for k in range(len(row[f"generated_{i}_action"])):
                generated_left_var = row[f"generated_{i}_action"][k]
                if label_left_action[0] == generated_left_var[0] and label_left_action[1] == generated_left_var[1]:
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


    results = pd.read_parquet(datapipeline_args.generated_processed)
   
    dataframe = pd.DataFrame(columns=["Node Type", "Node Type Accuracy", "Full Matching","Full Matching with Right Node Type", "Num Samples"])
    
    
    ### Node Type Accuracy
    node_type_equality = results.apply(node_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
    left_action_equality = results.apply(left_node_action_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
    
    print(len(node_type_equality))
    print(len(left_action_equality))
    newdf = pd.DataFrame({
        "nodetypeacc": node_type_equality,
        "left_action_acc": left_action_equality
    })
    newdf.to_csv("./metrics/baseline_details.csv")
    exit()

    node_type_accuracy= sum(node_type_equality) / len(node_type_equality)
    node_action_accuracy = sum(left_action_equality) / len(left_action_equality)
    
    property_with_right_node_type = results[node_type_equality == True]
    property_with_right_node_type_equality = property_with_right_node_type.apply(left_node_action_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
   
    property_with_right_node_type_accuracy = sum(property_with_right_node_type_equality) / len(property_with_right_node_type_equality)    
    
    dataframe.loc[len(dataframe)] = ["Global", node_type_accuracy, node_action_accuracy,property_with_right_node_type_accuracy, len(results)]

    ### Individual Accuracy 
    for node_type in node_type_list:
        results_individual_node_type = results[results["label_node_type"] == node_type]
        if len(results_individual_node_type) == 0:
            continue
        
        individual_node_type_equality = results_individual_node_type.apply(node_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
        individual_left_action_equality = results_individual_node_type.apply(left_node_action_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
        
        individual_property_with_right_node_type = results_individual_node_type[individual_node_type_equality == True]
        individual_property_with_right_node_type_equality = individual_property_with_right_node_type.apply(left_node_action_accuracy, axis=1, num_generated_sequences=datapipeline_args.num_generated_sequences)
    
        #print(f"Individual Node Type: {node_type} ({len(results_individual_node_type)} samples)")
        
        individual_node_type_accuracy= sum(individual_node_type_equality) / len(individual_node_type_equality)
        #print(f"{node_type} Accuracy (Node): {individual_node_type_accuracy}")
        
        individual_action_accuracy = sum(individual_left_action_equality) / len(individual_left_action_equality)
        #print(f"{node_type} Accuracy (Left Action): {individual_action_accuracy}\n")
        if len(individual_property_with_right_node_type) == 0:
            individual_property_with_right_node_type_accuracy = 0
        else:
            individual_property_with_right_node_type_accuracy = sum(individual_property_with_right_node_type_equality) / len(individual_property_with_right_node_type_equality)

        dataframe.loc[len(dataframe)] = [node_type, individual_node_type_accuracy, individual_action_accuracy, individual_property_with_right_node_type_accuracy, len(results_individual_node_type)]
    
    fileName = datapipeline_args.generated_processed.split("/")[-1]
    fileName = fileName.split(".")[0]

    if not os.path.isdir("metrics"):
        os.mkdir("metrics")


    dataframe.to_csv(f"metrics/{fileName}_metrics.csv", index=False)



    print(dataframe)
