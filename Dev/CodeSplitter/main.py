import os
import pandas as pd
import numpy as np
from msc_llms_context_retrieval.CodeSplitter import CodeSplitter
from BertClassifier import BertClassifier
import datasets
import torch
def load_file(file_path):
    return pd.read_parquet(file_path)


def get_dataset(code):
    splitter = CodeSplitter()
    categories, query_pos = splitter.split(code)
    print(categories)
    return categories, query_pos

def rank_snippets(snippets):
    bc = BertClassifier()
    return bc.classify(snippets)



def main(file, code_pos = 0):
    database = load_file(file)
    code =  database['text'].iloc[code_pos]
    label = database['label'].iloc[code_pos]

    categories, query_pos = get_dataset(code)

    categories.loc[len(categories)] = [label, -1, "label"]

    dataset = datasets.Dataset.from_pandas(categories)
    
    ranks = rank_snippets(dataset)
    print(ranks)
    
    scores = compareRankingsWithQuery(ranks, ranks[len(ranks)-1])
    
    value, indexes = torch.topk(torch.tensor(scores), 10)
    indexes = indexes.numpy().tolist()
    value = value.numpy().tolist()
    for query in query_pos:
        print(indexes)
        if query not in indexes:
            indexes.append(query)
            value.append(0)
    for i in range(len(indexes)):
        index = indexes[i]
        print("Rank: ", i)
        print("Score: ", value[i])
        print("Label: ", categories['category'].iloc[index])
        print("Code: \n", categories['code'].iloc[index])
        
    
    selected_code = categories.iloc[indexes].sort_values(by='pos')

    selected_code = selected_code[selected_code['pos'] != -1]
    
    print("\nSelected Code: ")
    for i, row in selected_code.iterrows():
        print(row['code'])
        


    
        

    
    


def compareRankingsWithQuery(ranks, query):
    scores = []

    for rank in ranks:
        scores.append(pytorch_cos_sim(rank, query))
    return scores


## Deep Seek Coder Help
def pytorch_cos_sim(embedding_1, embedding_2):
    # If embedding_1 is one-dimensional, unsqueeze it to be a batch of size 1
    if len(embedding_1.shape) == 1:
        embedding_1 = embedding_1.unsqueeze(0)
    
    # If embedding_2 is one-dimensional, unsqueeze it to be a batch of size 1
    if len(embedding_2.shape) == 1:
        embedding_2 = embedding_2.unsqueeze(0)
    
    # Normalize the embeddings
    embedding_1_normalized = torch.nn.functional.normalize(embedding_1, p=2, dim=1)
    embedding_2_normalized = torch.nn.functional.normalize(embedding_2, p=2, dim=1)
    
    # Compute the cosine similarity
    cos_sim = torch.mm(embedding_1_normalized, embedding_2_normalized.t()).item()
    
    return cos_sim


if __name__ == "__main__":
    dir = '..//msc_llms_context_retrieval//msc_llms_context_retrieval//datasets//logic//partial_test_python_v1_dataset.parquet'
    main(dir)
