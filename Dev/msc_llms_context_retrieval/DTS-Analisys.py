from cProfile import label
from operator import index
import numpy as np
import pandas as pd
import json
import Stats.TokenAnalisys as tkAnalisys
import Stats.TimeAnalisys as timeAnalisys

def table_accuracy(true, predicted):
    # If true (list) contains all elements of predicted (list), return 1, else, return 0
    if all(elem in predicted for elem in true):
        return 1
    else:
        return 0

def read_json(path):

    results = pd.read_json(path)
    # Transpose the DataFrame
    results = results.transpose()
    results = results.reset_index()
    results['num_tokens'] = pd.to_numeric(results['num_tokens'], errors='coerce')
    results['generation_time'] = pd.to_numeric(results['generation_time'], errors='coerce')
    
    # Get Num_tokens To Comparison
    num_tokens_sorted = results["num_tokens"].sort_values()

    
    # Get Information to compare times
    generation_time_valid = results["generation_time"] > 0
    num_tokens_to_time_graph = results["num_tokens"]
    generation_time_to_time_graph = results["generation_time"]


    return num_tokens_sorted, num_tokens_to_time_graph, generation_time_to_time_graph, generation_time_valid


def read_csv(path):
    pass    

if __name__ == "__main__":

    baseline_num_tokens_sorted, baseline_num_tokens, baseline_generation_time, baseline_generation_time_indexes = read_json("DTS-SQL-MAIN_RESULTS/predict_test_baseline.json")
    
    theirs_2nd_step_num_tokens_sorted, theirs_2nd_step_num_tokens, theirs_2nd_step_generation_time, theirs_2nd_step_generation_time_indexes = read_json("DTS-SQL-MAIN_RESULTS/predict_test_v2.json")
    indexes = theirs_2nd_step_num_tokens_sorted <= 111
    theirs_2nd_step_num_tokens_sorted.loc[indexes] = baseline_num_tokens_sorted.loc[indexes]
    
    ours_2nd_step_num_tokens_sorted, ours_2nd_step_num_tokens, ours_2nd_step_generation_time, ours_2nd_step_generation_time_indexes = read_json("DTS-SQL-MAIN_RESULTS/predict_test_step_model_v2.json")


    tkAnalisys.compareTokensPercentiles([baseline_num_tokens_sorted, theirs_2nd_step_num_tokens_sorted, ours_2nd_step_num_tokens_sorted], legend=["Baseline","Their two-step approach", "Our two-step approach"], limit=1000, )

    theirs_1st_step_results = pd.read_csv("DTS-SQL-MAIN_RESULTS/results-v2-step.csv")
    theirs_1st_step_num_tokens = theirs_1st_step_results["num_tokens"]
    theirs_1st_step_num_tokens_sorted = theirs_1st_step_results["num_tokens"].sort_values()
    theirs_1st_step_generation_time = theirs_1st_step_results["elapsed_time"]

    idx = baseline_generation_time_indexes & theirs_2nd_step_generation_time_indexes & ours_2nd_step_generation_time_indexes

    theirs_full_time = (theirs_1st_step_generation_time +  theirs_2nd_step_generation_time)

    for i in range(len(theirs_1st_step_generation_time)):
        if theirs_1st_step_generation_time[i] <= 0:
            theirs_1st_step_generation_time[i] = pd.NA

    


    ours_1st_step_results = pd.read_csv("DTS-SQL-MAIN_RESULTS/results-v2-step_step_model.csv")
    ours_1st_step_num_tokens = ours_1st_step_results["num_tokens"]
    ours_1st_step_num_tokens_sorted = ours_1st_step_results["num_tokens"].sort_values()
    ours_1st_step_generation_time = ours_1st_step_results["elapsed_time"]
    ours_full_time = (ours_1st_step_generation_time +  ours_2nd_step_generation_time)

    timeAnalisys.compareTime([baseline_num_tokens.loc[idx], theirs_2nd_step_num_tokens.loc[idx], ours_2nd_step_num_tokens.loc[idx]], [baseline_generation_time.loc[idx], theirs_full_time.loc[idx], ours_full_time.loc[idx]], legend=["Baseline","Their two-step approach", "Our two-step approach"])
 
    tkAnalisys.compareTokensPercentiles([theirs_1st_step_num_tokens_sorted, ours_1st_step_num_tokens_sorted], legend=["Their two-step approach", "Our two-step approach"], limit=1000, )
    timeAnalisys.compareTime([theirs_1st_step_num_tokens, ours_1st_step_num_tokens], [theirs_1st_step_generation_time, ours_1st_step_generation_time], legend=["Their two-step approach", "Our two-step approach"])
    
    
    
    ground_truth_results = pd.read_csv("DTS-SQL-MAIN_RESULTS/test_ground_thruth.csv")
    match = theirs_1st_step_results["num_tokens"] > 1000
    match2 = theirs_1st_step_results["num_tokens"] <= 1000
    
    match3 = ours_1st_step_results["num_tokens"] > 1000
    for i in range(len(match)):
        if match[i]:
            theirs_1st_step_results["tables"][i] = []
 
    # Use the mask to filter the rows in both DataFrames
    full_match = pd.DataFrame({
        "true": ground_truth_results["tables"],
        "predicted": theirs_1st_step_results["tables"]
    })
    full_match["accuracy"] = full_match.apply(lambda row : table_accuracy(row["true"], row["predicted"]), axis= 1)
    full_match.to_csv("DTS-SQL-MAIN_RESULTS/full_match_accuracy.csv")
    print(theirs_1st_step_num_tokens)
    full_match_accuracy = full_match["accuracy"].sum() / len(full_match)
    
    partial_match = pd.DataFrame({
        "true": ground_truth_results["tables"],
        "predicted": ours_1st_step_results["tables"]
    })
    partial_match["accuracy"] = partial_match.apply(lambda row : table_accuracy(row["true"], row["predicted"]), axis= 1)
    partial_match.to_csv("DTS-SQL-MAIN_RESULTS/partial_match_accuracy.csv")
    partial_match_accuracy = partial_match["accuracy"].sum() / len(full_match)


    print(f"Full Context Accuracy {full_match_accuracy}")
    print(f"Partial Context Accuracy {partial_match_accuracy}")

# Tentar perceber numeros finais

# Rever o relatóio
# Focar no Relatório
# 
# Fazer teste estatisticos
# 