
import pandas as pd
from sql_metadata import Parser

def get_columns(query):
    return Parser(query).tables



if __name__ == "__main__":
    test_file = pd.read_json("train/test.json")
    print(test_file)
    test_file["columns"] = test_file["SQL"].apply(get_columns)

    print(test_file["columns"])

    test_file.to_csv("test_ground_thruth.csv")