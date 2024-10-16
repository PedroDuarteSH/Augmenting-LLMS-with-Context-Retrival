import pandas as pd
import plotly.express as px
from .PlotColors import PlotColors
plot_colors = PlotColors()

def processNodeTypeLabel(nodetype : str):
    if nodetype[0] == "I":
        nodetype = nodetype[1:]

    if "Node" in nodetype or "node" in nodetype:
        nodetype = nodetype[:len(nodetype) - 4]
    return nodetype

CUSTOM_COLOR_SCHEME = [PlotColors.red()["normal"], PlotColors.blue()["normal"], PlotColors.green()["normal"], PlotColors.purple()["normal"]]

# Define the layout settings in a variable
common_layout = {
    'xaxis': {
        'tickangle': -90,  # Rotate labels by -45 degrees
        'tickfont': {'size': 14},
        'titlefont' : {'size' : 16}
    },
    'yaxis': {
        'tickfont': {'size': 14},
          'titlefont' : {'size' : 16}
    },
    'title': {
        'font': {'size': 20}  # Increase title font size
    },
    'legend': {
        'font': {'size': 14}  # Increase legend font size
    }
}


def node_accuracy_chart(full_dataframe):
    fig =  px.bar(full_dataframe, x="Node Type", y="Node Type Accuracy", color="Category", barmode="group", color_discrete_sequence=CUSTOM_COLOR_SCHEME)
    fig.update_layout(**common_layout)
    fig.show()


def full_matching_chart(full_dataframe):
    fig = px.bar(full_dataframe, x="Node Type", y="Full Matching", color="Category", barmode="group", color_discrete_sequence=CUSTOM_COLOR_SCHEME)
    fig.update_layout(**common_layout)
    fig.show()


def full_matching_chart_node_type_correct(full_dataframe):
    fig = px.bar(full_dataframe, x="Node Type", y="Full Matching with Right Node Type", color="Category", barmode="group", color_discrete_sequence=CUSTOM_COLOR_SCHEME)
    fig.update_layout(**common_layout)
    fig.show()

def compare_by_selected_token(dataframes, list_categories = ["Full Context", "Flow + Datamodel", "Flow + Imports", "Flow"]):
    if len(dataframes) != len(list_categories):
        raise Exception("The number of dataframes provided must be equal to the number of categories, and by the same order")
    for i, dataframe in enumerate(dataframes):
        dataframe["Category"] = list_categories[i]
    
    full_dataframe = pd.concat(dataframes)
    print(full_dataframe.columns)
    print(full_dataframe)
    title_node = "Node Type Accuracy per Category divided by Node Type (Discarding all elements where num tokens > 2048 considering Full Context)"
    title_full = "Full Properties Matching Accuracy per Category divided by Node Type (Discarding all elements where num tokens > 2048 considering Full Context)"
    node_accuracy_chart(full_dataframe)
    full_matching_chart(full_dataframe)

def compare_approaches(dataframes, legend = ["Baseline", "Two-Step Aproach"]):
    if len(dataframes) != len(legend):
        raise Exception("The number of dataframes must be equal to the mumber of categories, and by the same order")
    
    for i, dataframe in enumerate(dataframes):
        dataframe["Category"] = legend[i]

    title_node = "Node Type Accuracy per Approach divided by Node Type"
    title_full = "Full Properties Matching Accuracy per Approach divided by Node Type"
    title_full_node_type = "Full Properties Matching Accuracy per Approach divided by Node Type (When node type is correct)"
    full_dataframe = pd.concat(dataframes)
    full_dataframe["Node Type"] = full_dataframe["Node Type"].apply(processNodeTypeLabel)
    node_accuracy_chart(full_dataframe)
    full_matching_chart(full_dataframe)
    full_matching_chart_node_type_correct(full_dataframe)
