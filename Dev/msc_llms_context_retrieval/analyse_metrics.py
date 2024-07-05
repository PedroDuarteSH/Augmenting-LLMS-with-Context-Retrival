
import pandas as pd
import plotly.express as px

def node_accuracy_chart(full_dataframe):
    px.bar(full_dataframe, x="Node Type", y="Node Type Accuracy", color="Category", text="Num Samples", barmode="group",
           title="Node Type Accuracy per Category divided by Node Type (Discarding all elements where num tokens > 2048 considering Full Context)").show()

def full_matching_chart(full_dataframe):
    dataframe_without_end = full_dataframe[full_dataframe["Node Type"] != "<END>"]


    px.bar(dataframe_without_end, x="Node Type", y="Full Matching", color="Category", text="Num Samples", barmode="group",
           title="Full Properties Matching Accuracy per Category divided by Node Type (Discarding all elements where num tokens > 2048 considering Full Context)").show()

if __name__ == "__main__":
    # Load files
    full_context_metrics_file = "metrics/partial_test_python_v1_dataset_results_metrics.csv"
    full_context_metrics = pd.read_csv(full_context_metrics_file)
    full_context_metrics["Category"] = "Full Context"
    
    flow_dataset_metrics_file = "metrics/partial_test_python_v1_dataset_classes_flows_results_metrics.csv"
    flow_dataset_metrics = pd.read_csv(flow_dataset_metrics_file)
    flow_dataset_metrics["Category"] = "Flow and Dataset"


    flow_metrics_file = "metrics/partial_test_python_v1_dataset_flow_only_results_metrics.csv"
    flow_metrics = pd.read_csv(flow_metrics_file)
    flow_metrics["Category"] = "Flow Only"

    flow_imports_metrics_file = "metrics/partial_test_python_v1_dataset_flow_imports_results_metrics.csv"
    flow_imports_metrics = pd.read_csv(flow_imports_metrics_file)
    flow_imports_metrics["Category"] = "Flow and Imports"

    full_dataframe = pd.concat([full_context_metrics, flow_dataset_metrics, flow_metrics, flow_imports_metrics])
    ## Discard INRSendEmailNode 
    full_dataframe = full_dataframe[full_dataframe["Node Type"] != "INRSendEmailNode"]


    # Create a bar chart
    node_accuracy_chart(full_dataframe)
    full_matching_chart(full_dataframe)
    


