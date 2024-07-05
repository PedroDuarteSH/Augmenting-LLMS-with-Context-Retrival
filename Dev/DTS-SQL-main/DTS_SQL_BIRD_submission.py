import torch
import re
import sqlite3
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from torch import cuda  # noqa: F401
from transformers import StoppingCriteria
from tqdm import tqdm
from sql_metadata import Parser
import json
import numpy as np
import os
import gc
from accelerate.utils import release_memory
from peft import PeftModel
import time

BASE_DATASET_DIR = "./train/test.json"
BASE_DABATASES_DIR =  "./train/train_databases/"
OUTPUT_DIR = "predict_test_step_model.json"

model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

#loading the models
schema_model = AutoModelForCausalLM.from_pretrained(model_name,device_map = "auto")
schema_model = PeftModel.from_pretrained(schema_model, "final_checkpoint_Schema_linking", torch_dtype = torch.float16)
schema_model = schema_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

#helper functions
class EosListStoppingCriteriaSQL(StoppingCriteria):
    def __init__(self, eos_sequence = [6204, 185, 10897]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


class EosListStoppingCriteriaSchema(StoppingCriteria):
    def __init__(self, eos_sequence = [6204, 185, 10897]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def append_item(query,db_id, counter, output_dir, num_tokens, num_tokens_output):
    try:
        with open(output_dir, 'r') as json_file:
            data_dict = json.load(json_file)
    except FileNotFoundError:
        data_dict = {}
    item_value = query + "\t----- bird -----\t" + db_id + "\t" + str(num_tokens) + "\t" + str(num_tokens_output)
    data_dict[counter] = item_value
    with open(output_dir, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def remove_spaces(text):
  return re.sub(r'\s+', ' ', text)

def get_all_table_names(db_uri: str) -> list[str]:
    #print(db_uri)
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    conn.close()
    return [table_name[0] for table_name in table_names]

def get_table_schema_with_samples(
    db_uri: str, table_name: str, sample_limit: int = 0, columns_description: dict[str, str] = {}
) -> str:
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()

    # Fetch table schema
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns = cursor.fetchall()
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
    foreign_keys = cursor.fetchall()
    cursor.execute(f"PRAGMA index_list(`{table_name}`);")
    primary_key_indices = cursor.fetchall()
    primary_key_columns = []

    for index_info in primary_key_indices:
        index_name = index_info[1]
        cursor.execute(f"PRAGMA index_info(`{index_name}`);")
        index_columns = cursor.fetchall()
        primary_key_columns.extend(column[2] for column in index_columns)

    # Construct CREATE TABLE statement
    schema_str = f"CREATE TABLE `{table_name}` (\n"
    for column in columns:
        column_name = column[1]
        data_type = column[2]
        schema_str += f"  {column_name} {data_type}"
        if column_name in primary_key_columns:
            schema_str += " PRIMARY KEY"
        for foreign_key in foreign_keys:
            if column_name == foreign_key[3]:
                schema_str += f" REFERENCES {foreign_key[2]}({foreign_key[4]})"
        if column_name in columns_description:
            schema_str += f" -- '{columns_description[column_name]}'"

        schema_str += ",\n"
    schema_str = schema_str.rstrip(",\n")
    schema_str += "\n);\n"

    
    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {sample_limit};")
    sample_rows = cursor.fetchall()

    if len(sample_rows) > 0:
        schema_str += f"Sample rows from `{table_name}`:\n"
        for row in sample_rows:
            formatted_row = ", ".join(str(item) for item in row)
            schema_str += f"{formatted_row}\n"

    conn.close()
    return schema_str

def load_descriptions(db_path: str, table_name: str) -> list[str]:
    if not os.path.exists(f"{db_path}/database_description/{table_name}.csv"):
        return {}
    try:
        df = pd.read_csv(f"{db_path}/database_description/{table_name}.csv")
    except Exception:
        return {}
    if "column_description" not in df.columns or "value_description" not in df.columns:
        return {}
    columns_description = {}
    for index, row in df.iterrows():
        if np.nan != row["column_description"] and pd.notna(row["column_description"]):
            columns_description[row["original_column_name"]] = remove_spaces(row["column_description"])
            if np.nan != row["value_description"] and pd.notna(row["value_description"]):
                columns_description[row["original_column_name"]] += f" has values: ({remove_spaces(row['value_description'])})"
    return columns_description


def geenrate_sql(inputs, merged_model):
  output_tokens = merged_model.generate(inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria = [EosListStoppingCriteriaSQL()])
  num_tokens = len(output_tokens[0]) - len(inputs[0])
  
  return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True), num_tokens

def generate_schema(inputs, merged_model):
  output_tokens = merged_model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria = [EosListStoppingCriteriaSchema()])
  generated_tokens = output_tokens[0][len(inputs[0]):]

  non_padding_tokens = [token for token in generated_tokens if token != tokenizer.pad_token_id]
  num_tokens = len(non_padding_tokens)

  #print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
  return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True), num_tokens

def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))

if __name__ == "__main__":
    results = []
    df = pd.read_json(BASE_DATASET_DIR)
    sizes_first_step = []

    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Schema Linking"):
        db_id = row['db_id']
        query = row['SQL']
        question = row['question']
        if row['evidence'] != "" and row['evidence'] is not None:
            question += " Hint: " + row['evidence']
        db_uri = f"{BASE_DABATASES_DIR}/{db_id}/{db_id}.sqlite"
        db_path = f"{BASE_DABATASES_DIR}/{db_id}"
        table_names = get_all_table_names(db_uri)
        database_schema = ""
        for table_name in table_names:
            columns_description = load_descriptions(db_id, table_name)
            schema = get_table_schema_with_samples(db_uri, table_name, 0, columns_description)
            database_schema += schema + "\n"
        start = time.time()
        user_message = f"""Given the following SQL tables, your job is to determine the tables and columns that the question is referring to.
{database_schema}
####
Question: {question}
"""
        messages = [
            {"role": "user", "content": user_message.strip()}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,tokenize = True).to(schema_model.device)
        
        sizes_first_step.append(len(inputs[0]))
        
        if len(inputs[0]) > 3000:
            end = time.time()
            
            results.append({
                "question": "",
                "db_id": db_id,
                "query": "",
                "database_schema": "",
                "num_tokens":len(inputs[0]),
                "elapsed_time":-1,
                "num_tokens_output":-1,
                "tables":table_names
                
            })
            continue
            
        response, num_output_tokens = generate_schema(inputs, schema_model)
        if "Tables: " in response:
            response = response.split("Tables: ")[1]
        if ";" in response:
            response = response.split(";")[0]
        schema_linking_tables = re.sub(r'\s+', ' ', response).strip()
        #print(f"Predicted schema: {schema_linking_tables}")
        
        
        schema_linking_tables = schema_linking_tables.split(", ")
        end = time.time()
        database_schema = ""
        try:
            for table in schema_linking_tables:
                table = table.replace("**", "").replace("--", "").replace("'","").strip()
                database_schema += get_table_schema_with_samples(db_uri, table)
                database_schema += "\n"
        except Exception:
            database_schema = ""
            #print(f"Table not found {schema_linking_tables}")
            all_tables = get_all_table_names(db_uri)
            for table in all_tables:
                columns_description = load_descriptions(db_id, table)
                database_schema = get_table_schema_with_samples(db_uri, table, 0, columns_description)
                database_schema += "\n"
        #print(database_schema)
        
        generation_time = end-start
        
        results.append({
            "question": question,
            "db_id": db_id,
            "query": query,
            "database_schema": database_schema,
            "num_tokens":len(inputs[0]),
            "elapsed_time":generation_time,
            "num_tokens_output":num_output_tokens,
            "tables":schema_linking_tables
        })
        release_memory(schema_model)
    del schema_model
    
    #print(sizes_first_step)
    
    
    
    pd.DataFrame(results).to_csv("./results-v1-step.csv")
    sql_model = AutoModelForCausalLM.from_pretrained(model_name,device_map = "auto")
    sql_model = PeftModel.from_pretrained(sql_model, "final_checkpoint_sql", torch_dtype = torch.float16)
    sql_model = sql_model.merge_and_unload()
    
    
    
    for index, row in tqdm(enumerate(results), total=len(results), desc="Generating SQL"):
        query = row['query']
        db_id = row['db_id']
        question = row['question']
        database_schema = row['database_schema']
        user_message = f"""Given the following SQL tables, your job is to generate the Sqlite SQL query given the user's question.
Put your answer inside the ```sql and ``` tags.
{database_schema}
####
Question: {question}
"""
        messages = [
            {"role": "user", "content": user_message.strip()}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,tokenize = True).to(sql_model.device)
        
        response, num_tokens = geenrate_sql(inputs, sql_model)
        if ";" in response:
            response = response.split(";")[0]
        if "```sql" in response:
            response = response.split("```sql")[1]
        response = re.sub(r'\s+', ' ', response).strip()
        if "SELECT" not in response:
            response = "SELECT * FROM " + schema_linking_tables[0]
        print(f"Predicted: {response}")
        
        print(f"Gold: {query}")
        
       
        
        append_item(response, db_id, index, OUTPUT_DIR, len(inputs[0]), num_tokens)