# encoding=utf8
import os
import pdb
import sys
import json
import numpy as np
import argparse
import sqlite3
import pickle
import warnings
import multiprocessing as mp
from collections import OrderedDict
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
import fcntl

class EvaluateTool(object):
    def __init__(self, args):
        self.predict_test = []
        self.gold = []
        self.exec_result = []
        self.counter = 0
        self.pred = True
        self.idx_max = 0
    
   
    
    def result_callback(self, result):
        self.exec_result.append(result)
    
    
    def save_file(self):
        self.counter += 1
        if self.pred:
            output_dir = f"./sql_pred.json"
        else:
            output_dir = f"./sql_truth.json"
        with open(output_dir, 'w') as json_file:
            # Acquire exclusive lock on the file
            fcntl.flock(json_file.fileno(), fcntl.LOCK_EX)

            # Perform operations on the file
            json.dump(self.exec_result, json_file, indent=4)
            
            # Release the lock
            fcntl.flock(json_file.fileno(), fcntl.LOCK_UN)
            del json_file
            
    
    def execute_sql(self, sql, db_path):
        # Connect to the database
        conn = sqlite3.connect(db_path)
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

        return results
    
    def execute_model(self, sql, db_place, idx):
        try:
            result = func_timeout(20.0, self.execute_sql, args=(sql, db_place))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout',)]
        except Exception as e:
            # print('except:{}'.format(e))
            result = [(f'error',)]  # possibly len(query) > 512 or not executable

        result = {'sql_idx': idx, 'results': result}
        print(result)
        return result
    
    def run_sqls_parallel(self, sqls, db_places, num_cpus=1):
        
        pool = mp.Pool(processes=num_cpus)
          
        for i, sql in enumerate(sqls):
            if i < self.idx_max:
                continue
            print('*************** processing {}th sql ***************'.format(i))
            print(sql)
            pool.apply_async(self.execute_model, args=(sql, db_places[i], i), callback=self.result_callback).get()
            
            if i % 100 == 0:
                print("Saving Current Results")
                self.save_file()
        self.save_file()    
        pool.close()
        pool.join()
    
    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

    def compute_execution_accuracy(self, gt_results, predict_results):
        num_correct = 0
        num_queries = len(gt_results)
        mismatch_idx = []

        for i, result in enumerate(gt_results):
            if set(result['results']) == set(predict_results[i]['results']):
                num_correct += 1
            else:
                mismatch_idx.append(i)

        acc = (num_correct / num_queries) * 100

        return acc
    
    def flatten_sqls(self, golds):
        sqls = []
        # db_ids = []
        db_places = []
        for i, result_items in golds.iterrows():
            #print(result_items)
            sqls.append(result_items['SQL'])
            # db_ids.append(result_items['db_id'])
            db_places.append("train/train_databases" + '/' + result_items['db_id'] + '/' + result_items['db_id'] + '.sqlite')
        
        return sqls, db_places


    def evaluate(self, preds, golds):
        print("HERE")
        preds = [pred.split("\t", 1)[0].strip() for pred in preds.values()]
  
        
        
        gold_sqls, db_places = self.flatten_sqls(golds=golds)
        pred_sqls = preds
        
        
        if os.path.isfile("./sql_pred.json"):
            json = pd.read_json("./sql_pred.json")
    
            for _, elem in json.iterrows():
                elem = elem.to_dict() 
                del elem
            del json
    
        # # just for debugging:
        self.pred = True
        self.counter = 0
        self.run_sqls_parallel(pred_sqls, db_places, num_cpus=4)
        
        if os.path.isfile("./sql_pred.json"):
            json = pd.read_json("./sql_pred.json")
            print(len(json))
            for _, elem in json.iterrows():
                elem = elem.to_dict()
                self.exec_result.append(elem.copy())
                del elem
            del json
        
        pred_results = self.sort_results(self.exec_result)
        
        if os.path.isfile("./sql_truth.json"):
            json = pd.read_json("./sql_truth.json")
            for _, elem in json.iterrows():
                elem = elem.to_dict()
                self.exec_result.append(elem.copy())
                del elem
            del json
        
        self.exec_result = []
        self.pred = False
        self.counter = 0
        self.run_sqls_parallel(gold_sqls, db_places, num_cpus=4)
        gold_results = self.sort_results(self.exec_result)
        
        
        exec_accuracy = self.compute_execution_accuracy(gt_results=gold_results, predict_results=pred_results)
        exec_results = {'exec': exec_accuracy}
        return {**exec_results}

