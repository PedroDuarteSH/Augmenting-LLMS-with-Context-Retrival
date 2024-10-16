
import pandas as pd
from transformers import AutoTokenizer


def formatting_prompts_func(training_dataset):
  output_texts = []
  for i in range(len(training_dataset['question'])):
    question = training_dataset['question'][i]
    print(question)
    correct_tables = training_dataset['correct_tables'][i]
    correct_columns = training_dataset['correct_columns'][i]
    database_schema = training_dataset['database_schema'][i]
    all_tables = training_dataset['all_tables'][i]
    if correct_columns:
        correct_columns = ", ".join(set(correct_columns.split(", ")))
    correct_tables = ", ".join(set(correct_tables.split(", ")))
    user_message = f"""Given the following SQL tables, your job is to determine the tables that the question is referring to.
{all_tables}
###
Question: {question}
"""
    assitant_message = f"""
```SQL
-- Tables: {correct_tables} ;
```
"""
    messages = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assitant_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    exit()
  return output_texts

if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    print("This is the more.py file")
    data = pd.read_csv("train/filtered_finetuning_dataset.csv")
    print(data.head())
    
    formatted_prompts = formatting_prompts_func(data)
    