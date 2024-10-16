import pandas as pd
import utils
from sklearn import metrics
import matplotlib.pyplot as plt
def accuracy_node_type(row):
    if row["label_node_type"] == row[f"output_node_type"]:
        return 1
    return 0

def accuracy_context_type(row):
    if row["label_context_type"] == row[f"output_context_type"]:
        return 1
    return 0





if __name__ == "__main__":

    path = "msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_flow_only_encoded_results.parquet"

    dataframe = pd.read_parquet(path)
    print(dataframe.columns)
   
    elems = dataframe["num_tokens_input"] < 2048
    
    dataframe = dataframe[elems]

    equals = (dataframe["label_node_type"] == dataframe[f"output_node_type"]).sum()
    #equal_context = (dataframe["label_context_type"] == dataframe[f"output_context_type"]).sum()

    print("Total accuracy:")
    print(f"\tAccuracy node type: {equals/len(dataframe)}")
    #print(f"\tAccuracy context type: {equal_context/len(dataframe)}")

    print("\nAccuracy per node type:")
    for v, k in utils.nodetype2id().items():
        node_dataframes = dataframe[dataframe["label_node_type"] == k]
        print(f"Node type: {v} - {len(node_dataframes)}")
        equals = (node_dataframes["label_node_type"] == node_dataframes[f"output_node_type"]).sum()
        #equal_context = (node_dataframes["label_context_type"] == node_dataframes[f"output_context_type"]).sum()

        print(f"\tAccuracy node type: {equals/len(node_dataframes)}")
        #print(f"\tAccuracy context type: {equal_context/len(node_dataframes)}")
        print("\n")

    confusion_matrix = metrics.confusion_matrix(dataframe["label_node_type"], dataframe[f"output_node_type"], labels=range(14))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = utils.nodetype2id().keys())

    cm_display.plot(cmap="Blues", xticks_rotation="vertical")
    plt.show()

    