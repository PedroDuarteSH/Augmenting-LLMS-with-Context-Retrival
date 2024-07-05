import pandas as pd
from tqdm import tqdm
from utils.database_formatter import get_table_schema_with_samples, get_all_table_names
from utils.sql_regularizator import format_and_lowercase_sql_query
from utils.prompts import (
    sql_generation_prompt_token_counter,
    schema_linking_prompt_token_counter,
)
import os
from transformers import AutoTokenizer
from sql_metadata import Parser
from sklearn.model_selection import train_test_split

BASE_DATABASES_DIR = "train/train_databases"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
CONTEXT_WINDOW = 2048
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_spider_train_set():
    dataset_created = os.path.exists("train/test.json") and os.path.exists("train/validation.json") and os.path.exists("train/train_partition.json")
    
    
    
    if os.path.exists("train/train.json") and not dataset_created:
        df = pd.read_json("train/train.json")
    
        train, test = train_test_split(df, test_size= 0.2, random_state=42)
        train, validation = train_test_split(train, test_size= 0.1, random_state=42)
    
        test.to_json("train/test.json")
        validation.to_json("train/validation.json")
        train.to_json("train/train_partition.json")
    
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
    else:
        train = pd.read_json("train/train_partition.json")
        validation = pd.read_json("train/validation.json")
        test = pd.read_json("train/test.json")
        
    return train, validation, test




def create_sql_generation_correct_tables(dataset, question, query, db_uri):
    correct_tables = Parser(query).tables
    correct_columns = Parser(query).columns
    database_schema_filtered = ""
    for table in correct_tables:
        database_schema_filtered += get_table_schema_with_samples(db_uri, table)
        database_schema_filtered += "\n"
    database_schema = ""
    all_tables = get_all_table_names(db_uri)

    for table in all_tables:
        database_schema += get_table_schema_with_samples(db_uri, table)
        database_schema += "\n"
    #print(database_schema)
    
    if (
        schema_linking_prompt_token_counter(question, database_schema, correct_tables, correct_columns, tokenizer)
        <= CONTEXT_WINDOW
    ):
    
        dataset.append(
            {
                "db_id": db_uri.split("/")[-1].split(".")[0],
                "question": question,
                "query": query,
                "filtered_database_schema": database_schema_filtered,
                "database_schema": database_schema,
                "correct_tables": ", ".join(correct_tables),
                "correct_columns": ", ".join(correct_columns),
                "all_tables": ", ".join(all_tables),
            }
        )
    #print(correct_tables)
    return dataset


def create_full_sql_generation(
    dataset, question, query, db_uri, full_finetuning_errors
):
    database_schema = ""
    all_tables = get_all_table_names(db_uri)
    for table in all_tables:
        database_schema += get_table_schema_with_samples(db_uri, table)
        database_schema += "\n"
    if (
        sql_generation_prompt_token_counter(question, database_schema, query, tokenizer)
        <= CONTEXT_WINDOW
    ):
        dataset.append(
            {
                "db_id": db_uri.split("/")[-1].split(".")[0],
                "question": question,
                "query": query,
                "database_schema": database_schema,
            }
        )
    else:
        full_finetuning_errors += 1
    return dataset, full_finetuning_errors


if __name__ == "__main__":
    # Load Spider train set
    df, validation, test = load_spider_train_set()
    #df = df.sample(n=10, random_state=40).reset_index(drop=True)
  
    finetuning_dataset = []
    filtered_finetuning_dataset = []
    full_finetuning_errors = 0
    filtered_finetuning_errors = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        db_id = row["db_id"]
        question = row["question"]
        query = row["SQL"]
        query = format_and_lowercase_sql_query(query)
        db_uri = f"{BASE_DATABASES_DIR}/{db_id}/{db_id}.sqlite"
        all_tables = get_all_table_names(db_uri)
        try:
            filtered_finetuning_dataset = create_sql_generation_correct_tables(
                filtered_finetuning_dataset, question, query, db_uri
            )
        except Exception as e:
            print(e.args)
            filtered_finetuning_errors += 1
        finetuning_dataset, full_finetuning_errors = create_full_sql_generation(
            finetuning_dataset, question, query, db_uri, full_finetuning_errors
        )
    # Save finetuning dataset
    print(f"Full finetuning set errors: {full_finetuning_errors}")
    print(f"Filtered finetuning set errors: {filtered_finetuning_errors}")
    df = pd.DataFrame(finetuning_dataset)
    print(df)
    df.to_csv("train/full_finetuning_dataset.csv", index=False)
    df = pd.DataFrame(filtered_finetuning_dataset)
    df.to_csv("train/filtered_finetuning_dataset.csv", index=False)
    # Load Spider dev set
    #validation = load_spider_dev_set()
    #validation = validation.sample(n=10).reset_index(drop=True)
    validation_dataset = []
    validation_dataset_fromatted = []
    filtered_validation_dataset = []
    validation_set_errors = 0
    validation_set_formatted_errors = 0
    filtered_validation_set_errors = 0
    for index, row in tqdm(validation.iterrows(), total=len(validation)):
        db_id = row["db_id"]
        question = row["question"]
        query = row["SQL"]
        formatted_query = format_and_lowercase_sql_query(query)
        db_uri = f"{BASE_DATABASES_DIR}/{db_id}/{db_id}.sqlite"
        try:
            filtered_validation_dataset = create_sql_generation_correct_tables(
                filtered_validation_dataset, question, formatted_query, db_uri
            )
        except Exception:
            filtered_validation_set_errors += 1
        validation_dataset_fromatted, validation_set_formatted_errors = create_full_sql_generation(
            validation_dataset_fromatted,
            question,
            formatted_query,
            db_uri,
            validation_set_formatted_errors,
        )
        validation_dataset, validation_set_errors = create_full_sql_generation(
            validation_dataset, question, query, db_uri, validation_set_errors
        )
    print(f"Filtered validation set errors: {filtered_validation_set_errors}")
    print(f"Validation set formatted errors: {validation_set_formatted_errors}")
    print(f"Validation set errors: {validation_set_errors}")
    # Save validation dataset
    
    os.makedirs("validation", exist_ok=True)
    validation = pd.DataFrame(validation_dataset)
    validation.to_csv("validation/validation_dataset.csv", index=False)
    validation = pd.DataFrame(validation_dataset_fromatted)
    validation.to_csv("validation/validation_dataset_formatted.csv", index=False)
    validation = pd.DataFrame(filtered_validation_dataset)
    validation.to_csv("validation/filtered_validation_dataset.csv", index=False)
    
    test_dataset = []
    test_dataset_fromatted = []
    filtered_test_dataset = []
    test_set_errors = 0
    test_set_formatted_errors = 0
    filtered_test_set_errors = 0
    for index, row in tqdm(test.iterrows(), total=len(test)):
        db_id = row["db_id"]
        question = row["question"]
        query = row["SQL"]
        formatted_query = format_and_lowercase_sql_query(query)
        db_uri = f"{BASE_DATABASES_DIR}/{db_id}/{db_id}.sqlite"
        try:
            filtered_test_dataset = create_sql_generation_correct_tables(
                filtered_test_dataset, question, formatted_query, db_uri
            )
        except Exception:
            filtered_test_set_errors += 1
        test_dataset_fromatted, test_set_formatted_errors = create_full_sql_generation(
            test_dataset_fromatted,
            question,
            formatted_query,
            db_uri,
            test_set_formatted_errors,
        )
        test_dataset, test_set_errors = create_full_sql_generation(
            test_dataset, question, query, db_uri, test_set_errors
        )
    print(f"Filtered test set errors: {filtered_test_set_errors}")
    print(f"Test set formatted errors: {test_set_formatted_errors}")
    print(f"Test set errors: {test_set_errors}")
    # Save test dataset
    
    os.makedirs("test", exist_ok=True)
    test = pd.DataFrame(test_dataset)
    test.to_csv("test/test_dataset.csv", index=False)
    test = pd.DataFrame(test_dataset_fromatted)
    test.to_csv("test/test_dataset_formatted.csv", index=False)
    test = pd.DataFrame(filtered_test_dataset)
    test.to_csv("test/filtered_test_dataset.csv", index=False)
