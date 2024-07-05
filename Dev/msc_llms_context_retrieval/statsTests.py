import pandas as pd
import scipy as sp
import os
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as stats

import numpy as np
def load_dataset_by_folder(path):
    listfolder = os.listdir(path)
    full_dataset = pd.DataFrame()
    for file in listfolder:
        print(file)
        dataset = pd.read_parquet(os.path.join(path, file))
        full_dataset = pd.concat([full_dataset, dataset])
    return full_dataset

def logic_problem():
    baseline_dataset = pd.read_parquet("./msc_llms_context_retrieval/datasets/logic/baseline_results_other.parquet")
    baseline_metrics = pd.read_csv("metrics/baseline_details.csv")
    na_values = baseline_dataset["generation_time"].notna()
    
    
    
    two_step_dataset = load_dataset_by_folder("msc_llms_context_retrieval/datasets/logic/node(context)_2step_model_end2end_test.parquet")
    two_step_metrics = pd.read_csv("metrics/2_step_details.csv")
    na_values2 = two_step_dataset["generation_time"].notna()

    na_values = na_values & na_values2
    baseline_time = baseline_dataset["generation_time"].loc[na_values]
    print(np.mean(baseline_time))
    two_step_time = two_step_dataset["generation_time"].loc[na_values]
    print(np.mean(two_step_time))
    #print(baseline_metrics)
    #print(two_step_metrics)



    # Calculate counts for each category
    a = sum((np.array(baseline_metrics["nodetypeacc"]) == 1) & (np.array(two_step_metrics["nodetypeacc"]) == 1))
    b = sum((np.array(baseline_metrics["nodetypeacc"]) == 1) & (np.array(two_step_metrics["nodetypeacc"]) == 0))
    c = sum((np.array(baseline_metrics["nodetypeacc"]) == 0) & (np.array(two_step_metrics["nodetypeacc"]) == 1))
    d = sum((np.array(baseline_metrics["nodetypeacc"]) == 0) & (np.array(two_step_metrics["nodetypeacc"]) == 0))

    # Create the contingency table
    contingency_table = np.array([[a, b], [c, d]])
    print(contingency_table)
    result = mcnemar(contingency_table, exact=True)
    print(f'Statistic: {result.statistic}, p-value: {result.pvalue}')

    stat1, p1 = stats.shapiro(baseline_time)
    stat2, p2 = stats.shapiro(two_step_time)
    print(f'Model 1: Shapiro-Wilk statistic={stat1}, p-value={p1}')
    print(f'Model 2: Shapiro-Wilk statistic={stat2}, p-value={p2}')

    statistic, p_value = stats.wilcoxon(baseline_time, two_step_time)
    print(f'Wilcoxon signed-rank test: statistic={statistic}, p-value={p_value}')




def DTS_SQL_problem():
    full_dataset_results = pd.read_csv("DTS-SQL-MAIN_RESULTS/results-v1-step.csv")
    full_match_accuracy = pd.read_csv("DTS-SQL-MAIN_RESULTS/full_match_accuracy.csv")
    full_dataset_generation_time = full_dataset_results["elapsed_time"]
    valid_values = full_dataset_generation_time > 0

    partial_dataset_results = pd.read_csv("DTS-SQL-MAIN_RESULTS/results-v1-step_step_model.csv")
    partial_match_accuracy = pd.read_csv("DTS-SQL-MAIN_RESULTS/partial_match_accuracy.csv")

    match = full_dataset_results["num_tokens"] > 1000
    
    match2 = full_dataset_results["num_tokens"] <= 1000
   
    for i in range(len(match)):
        print(match[i])
        if match[i]:
            full_match_accuracy["accuracy"][i] = 0

    a = sum((np.array(full_match_accuracy["accuracy"]) == 1) & (np.array(partial_match_accuracy["accuracy"]) == 1))
    b = sum((np.array(full_match_accuracy["accuracy"]) == 1) & (np.array(partial_match_accuracy["accuracy"]) == 0))
    c = sum((np.array(full_match_accuracy["accuracy"]) == 0) & (np.array(partial_match_accuracy["accuracy"]) == 1))
    d = sum((np.array(full_match_accuracy["accuracy"]) == 0) & (np.array(partial_match_accuracy["accuracy"]) == 0))

    # Create the contingency table
    contingency_table = np.array([[a, b], [c, d]])
    print(contingency_table)
    result = mcnemar(contingency_table, exact=True)
    print(f'Statistic: {result.statistic}, p-value: {result.pvalue}')

    partial_dataset_generation_time = partial_dataset_results["elapsed_time"]
    valid_values_partial = partial_dataset_generation_time.notna()

    valid_values = valid_values & valid_values_partial
    stat1, p1 = stats.shapiro(full_dataset_generation_time.loc[valid_values])
    stat2, p2 = stats.shapiro(partial_dataset_generation_time.loc[valid_values])
    print(f'Model 1: Shapiro-Wilk statistic={stat1}, p-value={p1}')
    print(f'Model 2: Shapiro-Wilk statistic={stat2}, p-value={p2}')

    statistic, p_value = stats.wilcoxon(full_dataset_generation_time.loc[valid_values], partial_dataset_generation_time.loc[valid_values])
    print(f'Wilcoxon signed-rank test: statistic={statistic}, p-value={p_value}')


if __name__ == "__main__":
    #logic_problem()
    DTS_SQL_problem()


    