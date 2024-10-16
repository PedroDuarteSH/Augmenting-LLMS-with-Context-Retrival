
import pandas as pd
import json
import plotly
import plotly.express as px
from transformers import AutoTokenizer


def bar_plot_token_details(y, imports, imports_values, imports_text, dataset, dataset_values, dataset_text, flow, flow_values, flow_text, query, query_values, query_text):
    fig = plotly.graph_objs.Figure()


    if len(query) > 0:
        fig.add_bar(x=query, y=query_values, name="Query", text=query_text, hoverinfo="text")
    if len(imports) > 0:
        fig.add_bar(x=imports, y=imports_values, name="Import", text=imports_text, hoverinfo="text")
    if len(dataset) > 0:
        fig.add_bar(x=dataset, y=dataset_values, name="Dataset", text=dataset_text, hoverinfo="text")
    if len(flow) > 0:
        fig.add_bar(x=flow, y=flow_values, name="Flow", text=flow_text, hoverinfo="text")

    fig.update_traces(textposition='outside')
    fig.show()

def box_plot(attention_metrics, node_type):
    # Remove rows with 0 tokens
    attention_metrics = attention_metrics[attention_metrics["num_tokens"] > 0]
    fig = px.box(attention_metrics, x="category", y="attention_avg", points="all", title=f"Attention average per category ({node_type})", labels={"attention_avg": "Average attention", "category": "Category"})
    fig.show()


if __name__ == "__main__":
    node_type = "IAggregateNode"
    path = f"msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_attention_nodes.parquet/{node_type}"
   
    tokenizer_dir = "./model/logic_flows_finetune_v2/model"
    dataset = pd.read_parquet(path)
    print(dataset.head())
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side="right")
    
    
    attention_metrics = pd.DataFrame(columns=["category", "num_tokens", "attention_sum", "attention_avg"])
    
    for encoded_mapping in dataset["encoder_attentions"]:
        attentionMapping = json.loads(encoded_mapping)
        processed_attention_mapping = {}
    
    
        # Remove the first and last token
        attentionMapping.pop(list(attentionMapping.keys())[0])
        attentionMapping.pop(list(attentionMapping.keys())[-1])
    
        query = []
        start_dataset = 0
        start_flow = 0
        y = list(attentionMapping.values())
        x = list(attentionMapping.keys())
        for i in range(len(x)):
            index = x[i].rfind("%")
            x[i] = x[i][:index]
            
            if x[i] == "@" and start_dataset == 0:
                start_dataset = i
                
            if x[i] == "var" and start_flow == 0:
                start_flow = i

            if "<extra_id" in x[i]:
                query.append(i)
        
        if start_dataset == 0:
            start_dataset = start_flow
        
        query_values = [y[i] for i in query]
        
        imports = list(range(0,start_dataset))
        imports_values = [y[i] for i in imports]
        
        
        dataset = list(range(start_dataset, start_flow))
        dataset = [i for i in dataset if y[i] != 0]
        dataset_values = [y[i] for i in dataset]
        
        flow = [i for i in range(start_flow, len(x)) if i not in query]        
        flow_values = [y[i] for i in flow]
        
        attention_metrics.loc[len(attention_metrics)] = ["imports", len(imports), sum(imports_values), sum(imports_values)/len(imports) if len(imports) > 0 else 0]
        attention_metrics.loc[len(attention_metrics)] = ["dataset", len(dataset), sum(dataset_values), sum(dataset_values)/len(dataset) if len(dataset) > 0 else 0]
        attention_metrics.loc[len(attention_metrics)] = ["flow", len(flow), sum(flow_values), sum(flow_values)/len(flow) if len(flow) > 0 else 0]
        #attention_metrics.loc[len(attention_metrics)] = ["query", len(query), sum(query_values), sum(query_values)/len(query) if len(query) > 0 else 0]
    
    
    query_text = [x[i] for i in query]
    query_values = [y[i] for i in query]
    
    imports_text = [x[i] for i in imports]
    imports_values = [y[i] for i in imports]
    
    flow_text = [x[i] for i in flow]
    flow_values = [y[i] for i in flow]
    
    dataset_text = [x[i] for i in dataset]
    dataset_values = [y[i] for i in dataset]
        
    
    
    #
    #bar_plot_token_details(y, imports, imports_values, imports_text, dataset, dataset_values, dataset_text, flow, flow_values, flow_text, query, query_values, query_text)
    box_plot(attention_metrics, node_type)