from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, DataCollatorWithPadding, HfArgumentParser, EarlyStoppingCallback
from params import TrainingLogicDatasetArguments
import torch
from datasets import load_dataset
from pathlib import Path
import os
import utils
import pandas as pd
from aim.hugging_face import AimCallback
from aim.sdk.objects.plugins.dvc_metadata import DvcData
import numpy as np
import evaluate
ROOT_FOLDER: Path = Path(__file__).parent

# the maximum number of validation instances
MAX_VALIDATION_SIZE: int = 5000

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class NodeTypeClassificationTrainer(Trainer):
    def __init__(self, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = torch.tensor(weights).to("cuda")
        self.softmax = torch.nn.functional.softmax
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.num_labels = len(weights)


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = self.softmax(logits)
        loss = self.loss_fcn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
        


def compute_weights(dataset):
    data = pd.Series(dataset["node_type_int"]).value_counts().sort_index()
    weights = len(dataset) / (data * len(data)) 
    weights = weights.to_list() 
    return weights

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":

    parser = HfArgumentParser([TrainingLogicDatasetArguments, TrainingArguments])
    datapipeline_args: TrainingLogicDatasetArguments = parser.parse_args_into_dataclasses()[0]
    training_args: TrainingArguments = parser.parse_args_into_dataclasses()[1]

    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.tokenizer_dir, padding_side="right")

    config = AutoConfig.from_pretrained('nomic-ai/nomic-bert-2048', trust_remote_code=True,  num_labels = len(utils.nodetype2id()), vocab_size=len(tokenizer))
    model = AutoModelForSequenceClassification.from_pretrained("nomic-ai/nomic-bert-2048", config=config, trust_remote_code=True, strict=False, num_labels = len(utils.nodetype2id()))

    

    train_dataset_path = Path(datapipeline_args.tokenized_data)
    print(train_dataset_path)
    if train_dataset_path.is_dir():
        dataset = load_dataset(datapipeline_args.train_data, split="train", num_proc=os.cpu_count())
    else:
        dataset = load_dataset(
            "parquet",
            data_files=datapipeline_args.tokenized_data,
            split="train",
            num_proc=os.cpu_count(),
        )


  
     # select columns that matter for train
    print(dataset.column_names)
    total_instances = len(dataset)
    # filter instances that do not fit in the total number of tokens we are enabling
    weights = compute_weights(dataset)
    
    dataset = dataset.rename_column("node_type_int", "labels")

    

    if datapipeline_args.max_number_of_train_tokens is not None:
        print(f"Filtering instances with more than {datapipeline_args.max_number_of_train_tokens} tokens")
        dataset = dataset.filter(
            lambda x, max_number_of_train_tokens = datapipeline_args.max_number_of_train_tokens: (
                len(x["input_ids"]) <= max_number_of_train_tokens |
                x["labels"] > len(utils.nodetype2id())
            ),
            num_proc=os.cpu_count(),
        )

        print(f"Discarded (#tokens): {total_instances-len(dataset)}/{total_instances}")
        total_instances = len(dataset)
    
    print(dataset)
    dataset = dataset.select_columns(["input_ids", "labels"])


    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=2048)

    aim_callback = AimCallback(experiment="logic_flow_encoder_train")
    aim_callback.experiment["dvc_info"] = DvcData(ROOT_FOLDER)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=datapipeline_args.early_stopping_patience)
    if datapipeline_args.validation_frac:
        validation_size = int(len(dataset) * datapipeline_args.validation_frac)
        validation_size = min(MAX_VALIDATION_SIZE, validation_size)

        dataset = dataset.train_test_split(test_size=validation_size, seed=training_args.seed)

    trainer = NodeTypeClassificationTrainer(
        model=model,
        tokenizer=tokenizer,   
        args=training_args,
        train_dataset=dataset["train"] if datapipeline_args.validation_frac else dataset,
        eval_dataset=dataset["test"] if datapipeline_args.validation_frac else None,
        data_collator=data_collator,
        callbacks=[aim_callback, early_stopping_callback],
        weights=weights,
        compute_metrics=compute_metrics
    )


    # Train the model
    trainer.train()