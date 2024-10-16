import pandas as pd
import json
import io

from evaluator import EvaluateTool

BASE_DATASET_DIR = "./train/test.json"
PREDICT_FILE = "./predict_test.json"
if __name__ == "__main__":


    gold = pd.read_json(BASE_DATASET_DIR)
    # Load the JSON file
    with open(PREDICT_FILE, 'r') as f:
        predict_dict = json.load(f)
  
    ev = EvaluateTool(args=[])
    acc = ev.evaluate(predict_dict, gold)
    print(acc)
    # Write JSON file
    with io.open('./data.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(acc,
                        indent=4, sort_keys=True,
                        separators=(',', ': '), ensure_ascii=False)
        outfile.write(str(str_))
