from transformers import BertTokenizer, BertModel
import datasets
import torch

class BertClassifier:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        pass


    def classify(self, snippets: datasets.Dataset):
        encoding = self.tokenizer(snippets['code'], padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = self.model(**encoding)

        last_hidden_state = output.last_hidden_state
        
        input_representations = torch.mean(last_hidden_state, dim=1)


        return input_representations   


