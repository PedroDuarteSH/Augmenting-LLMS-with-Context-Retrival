import pandas as pd
from utils import process_labels
import os

if __name__ == "__main__":
    dataframe = pd.read_parquet("msc_llms_context_retrieval/datasets/logic/partial_test_python_v1_dataset_classes_flows_results.parquet")
    dataframe_text = pd.read_parquet("msc_llms_context_retrieval/datasets/logic/partial_test_python_v1_dataset_flow_imports.parquet")


    
    dataframe = dataframe[dataframe["label_node_type"] == "IAggregateNode"]


    print(dataframe.iloc[10])
    dataframe_text = dataframe_text.loc[dataframe_text["label"].str.startswith("IAggregateNode")]

    print(dataframe_text.iloc[10]["text"])
    print(dataframe_text.iloc[10]["label"])
    num_generated_sequences = 5
    dict_hit_rate_by_action = {}
    dict_others = []
    for item in dataframe.iterrows():
        if item[1]["label_action"] is None:
            continue
        for label_action in item[1]["label_action"]:
            if label_action[0] not in dict_hit_rate_by_action:
                ### [0] is the count, [1] is the hit rate on generated sequences
                dict_hit_rate_by_action[label_action[0]] = [1, 0]
            else:
                dict_hit_rate_by_action[label_action[0]][0] += 1
            for i in range(num_generated_sequences):
                if(item[1][f"generated_{i}_action"] is None):
                    continue
                for generated_action in item[1][f"generated_{i}_action"]:
                    action, value = label_action
                    gen_action, gen_value = generated_action
                    if generated_action[0] == label_action[0]:
                        if generated_action[1] == label_action[1]:
                            dict_hit_rate_by_action[label_action[0]][1] += 1
                            break
                    
    for i in range(num_generated_sequences):
        for generated_action in item[1][f"generated_{i}_action"]:
            if generated_action[0] not in dict_hit_rate_by_action:
                dict_others.append(generated_action[0])
    dict_hit_rate_by_action_sorted = {k: v for k, v in sorted(dict_hit_rate_by_action.items(), key=lambda item: item[1][0], reverse=True)}
    
    d = pd.DataFrame(columns=["action", "count", "hit_rate", "accuracy"])
    for index, item in enumerate(dict_hit_rate_by_action_sorted.items()):
        row = [item[0], item[1][0], item[1][1]]
        d.loc[len(d)] = [*row, (item[1][1]/item[1][0] if item[1][0] > 0 else 0)]
    os.makedirs("aggregate", exist_ok=True)
    d.to_csv("aggregate/aggregate_action_hit_rate_classes_flows.csv", index=False)
    
            