from scipy import datasets
from torch import ge
import Stats.TokenAnalisys as tkAnalisys
import Stats.TimeAnalisys as tmAnalisys
import Stats.LogicAnalisys as loAnalisys
import pandas as pd
import os
import pyarrow.parquet as pq

def load_dataset_by_folder(path):
    listfolder = os.listdir(path)
    full_dataset = pd.DataFrame()
    for file in listfolder:
        print(file)
        dataset = pd.read_parquet(os.path.join(path, file))
        return dataset
        full_dataset = pd.concat([full_dataset, dataset])
    return full_dataset

def compareBothApproaches():
    baseline_dataset = pd.read_parquet("./msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_results.parquet")
    baseline_metrics = pd.read_csv("metrics/baseline_results_other_metrics.csv")

    
    two_step_dataset = load_dataset_by_folder("msc_llms_context_retrieval/datasets/logic/node(context)_2step_model_end2end_test.parquet")
    two_step_metrics = pd.read_csv("metrics/node(context)_2step_model_end2end_test_metrics.csv")
    


    num_tokens_baseline = baseline_dataset["num_tokens_input"]
    generation_time_baseline = baseline_dataset["generation_time"]
    

    num_tokens_two_step_sorted = two_step_dataset["num_tokens_input_properties"].sort_values()
    num_tokens_two_step = two_step_dataset["num_tokens_input_properties"]
    generation_time_two_step = two_step_dataset["generation_time"]


    #tkAnalisys.plotTokensPercentil(num_tokens_baseline)

    #tkAnalisys.plotTokensPercentil(num_tokens_two_step_sorted)
    
    #

    #tmAnalisys.plotTime(num_tokens=num_tokens_two_step, generation_time=generation_time_two_step)

    #tmAnalisys.plotTime(num_tokens=num_tokens_baseline, generation_time=generation_time_baseline)
    
    #tkAnalisys.compareTokensPercentiles([num_tokens_baseline, num_tokens_two_step_sorted], legend=["Baseline", "Two Step Model"])
    #tmAnalisys.compareTime([num_tokens_baseline, num_tokens_two_step], [generation_time_baseline, generation_time_two_step ], legend=["Baseline", "Two Step Model"])
    loAnalisys.compare_approaches([baseline_metrics, two_step_metrics])

def comparePartialContext():
    full_context_metrics_file = "metrics/partial_test_python_v1_dataset_results_metrics.csv"
    full_context_metrics = pd.read_csv(full_context_metrics_file)

    flow_datamodel_metrics_file = "metrics/partial_test_python_v1_dataset_classes_flows_results_metrics.csv"
    flow_datamodel_metrics = pd.read_csv(flow_datamodel_metrics_file)

    flow_metrics_file = "metrics/partial_test_python_v1_dataset_flow_only_results_metrics.csv"
    flow_metrics = pd.read_csv(flow_metrics_file)

    flow_imports_metrics_file = "metrics/partial_test_python_v1_dataset_flow_imports_results_metrics.csv"
    flow_imports_metrics = pd.read_csv(flow_imports_metrics_file)
    
    loAnalisys.compare_by_selected_token([full_context_metrics, flow_datamodel_metrics, flow_imports_metrics, flow_metrics])




if __name__ == "__main__":
    comparePartialContext()
    compareBothApproaches()