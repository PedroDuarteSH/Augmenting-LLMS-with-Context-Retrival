import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import utils
if __name__ == "__main__":
    path = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_encoder_embedding.parquet"
    #path_2 = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only.parquet"

    dataset = pd.read_parquet(path)
    
    
    
    print(dataset["label"].value_counts())
    dataset.dropna(subset=["embedding"], inplace=True)
    dataset = dataset.reset_index(drop=True)
    embeddings = torch.tensor(dataset["embedding"])
    test_path = "msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_flow_only.parquet/5a91ea205cd441b485773eccccd13be0_000000.parquet"

    test_dataset = pd.read_parquet(test_path)
    node_types = test_dataset["label"].unique()

    print(test_dataset)


    checkpoint = "codesage/codesage-small"
    device = "cpu"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    count = 0
    num_right = 0
    for i, element in test_dataset.iterrows():
        inputs = tokenizer.encode(element["text"], return_tensors="pt").to(device)
      
        if  len(inputs[0]) >= 2048:
            print("Error Processing", i, element["text"][:50] + "...")
            continue
       
        output_with_pooling = model(inputs)
        last_hidden_state = output_with_pooling.last_hidden_state
        # Apply pooling operation, for example, mean pooling
        pooled_output = torch.mean(last_hidden_state, dim=1).squeeze().detach().numpy()
        
        #print(pooled_output)
        # Assuming dataset["embedding"] is a list of embedding lists
       
        
        

        # Assuming pooled_output is already a tensor
        pooled_output = torch.tensor(pooled_output)

        # Compute cosine similarity
        # The unsqueeze(0) adds a batch dimension to pooled_output for broadcasting
        dataset["similarity"] = torch.nn.functional.cosine_similarity(embeddings, pooled_output.unsqueeze(0), dim=1)
        dataset = dataset.sort_values(by="similarity", ascending=False)
        closer = dataset.iloc[:5]
        
        
        #print(closer["label"].value_counts())
        
        
        
        node_type = element["label"].split("(")[0]
        print("\n")
        print(node_type)
        print(closer[['label', 'similarity']])
        if node_type in closer["label"].values:
            num_right += 1
        count += 1
        if count == 10:
            break
        
    print(num_right/count)