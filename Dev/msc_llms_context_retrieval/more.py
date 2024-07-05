import pandas as pd
import re
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import os

if __name__ == "__main__":
    path = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_encoder.parquet/741c81ef0b134525bfc8eb4eed7147da_000000.parquet"
    output = "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_encoder_embedding.parquet/741c81ef0b134525bfc8eb4eed7147da_000000.parquet"
    
    os.makedirs("msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_encoder_embedding.parquet", exist_ok=True)
    
    checkpoint = "codesage/codesage-small"
    device = "cpu"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


    dataframe = pd.read_parquet(path)
    embedding = [None] * len(dataframe)
    for i, element in dataframe.iterrows():
        inputs = tokenizer.encode(element["text"], return_tensors="pt").to(device)
      
        if  len(inputs[0]) >= 2048:
            print("Error Processing", i, element["text"][:50] + "...")
            continue
       
        output_with_pooling = model(inputs)
        last_hidden_state = output_with_pooling.last_hidden_state
        # Apply pooling operation, for example, mean pooling
        pooled_output = torch.mean(last_hidden_state, dim=1).squeeze().detach().numpy()
        
        embedding[i] = pooled_output

    dataframe["embedding"] = embedding
    print(dataframe)

    dataframe.to_parquet(output)
