import os
def create_parameters_string(parameters):
    parameters_string = ""
    for key in parameters:
        parameters_string += f"--{key} {parameters[key]} "
    return parameters_string


if __name__ == "__main__":
    """
    parameters = { 
        "model_type": "enc-dec",
        "training_type": "masking",
        "tokenizer_name": 'Salesforce/codet5-small',
        "model_name": 'Salesforce/codet5-small',
        "max_number_of_train_tokens": 2048,
        "max_gen_length": 256,
        "tokenizer_dir": "tokenizer",
        "train_data": "./msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_tokenized.parquet",
        "tokenized_data": "./msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_tokenized.parquet",

        "validation_frac": 0.02,
        "output_dir": './model/logic_flows_finetune_v2/model',
        "learning_rate": 5e-5,
        "weight_decay": 0.05,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_steps": 50000,
        "warmup_steps": 100,
        "fp16": False,
        "save_total_limit": 4,
        "eval_steps": 3000,
        "save_steps": 3000,
        "logging_steps": 3000,
        "seed": 42,
        "data_seed": 42,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "early_stopping_patience": 3,
        "evaluation_strategy":"steps"
    }
    """
    parameters = { 
        "model_type": "enc",
        "training_type": "masking",
        "tokenizer_name": 'nomic-ai/nomic-bert-2048',
        "model_name": 'nomic-ai/nomic-bert-2048',
        "max_number_of_train_tokens": 2048,
        "max_gen_length": 256,
        "tokenizer_dir": "encoder_tokenizer",
        "train_data": "./msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only_encoded_node_only_tokenized.parquet",
        "tokenized_data": "./msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only_encoded_node_only_tokenized.parquet",
        "num_train_epochs" : 1,
        "validation_frac": 0.02,
        "output_dir": './model/logic_flows_finetune_encoder_node_class_only_v3/model',
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 16,
        #"max_steps": 50000,
        "warmup_steps": 100,
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "save_total_limit": 4,
        "eval_steps": 3000,
        "save_steps": 3000,
        "logging_steps": 3000,
        "seed": 42,
        "data_seed": 42,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "early_stopping_patience": 10,
        "evaluation_strategy":"steps"
    }


    params = create_parameters_string(parameters)
    
    #execute = f"python finetune_codet5.py {params}"
    execute = f"python finetune_bert.py {params}"
    os.system(execute)





