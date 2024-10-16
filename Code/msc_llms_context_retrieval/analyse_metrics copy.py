
import pandas as pd
import plotly.express as px

def aggregate_properties_chart(full_dataframe):

    px.bar(full_dataframe, x="action", y="accuracy", color="Category", text="count", barmode="group",
           title="Aggregate Individual Properties Matching").show()

if __name__ == "__main__":
    # Load files
    full_context_metrics_file = "aggregate/aggregate_action_hit_rate.csv"
    full_context_metrics = pd.read_csv(full_context_metrics_file)
    full_context_metrics["Category"] = "Full Context"
    
    flow_dataset_metrics_file = "aggregate/aggregate_action_hit_rate_classes_flows.csv"
    flow_dataset_metrics = pd.read_csv(flow_dataset_metrics_file)
    flow_dataset_metrics["Category"] = "Flow and Dataset"


    flow_metrics_file = "aggregate/aggregate_action_hit_rate_flow_only.csv"
    flow_metrics = pd.read_csv(flow_metrics_file)
    flow_metrics["Category"] = "Flow Only"

    flow_imports_metrics_file = "aggregate/aggregate_action_hit_rate_flow_imports.csv"
    flow_imports_metrics = pd.read_csv(flow_imports_metrics_file)
    flow_imports_metrics["Category"] = "Flow and Imports"

    full_dataframe = pd.concat([full_context_metrics, flow_dataset_metrics, flow_metrics, flow_imports_metrics])


    aggregate_properties_chart(full_dataframe)
    


