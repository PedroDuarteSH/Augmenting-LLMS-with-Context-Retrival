"""Training script."""

import os
from pathlib import Path

from aim.hugging_face import AimCallback
from aim.sdk.objects.plugins.dvc_metadata import DvcData
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from params import TrainingLogicDatasetArguments

ROOT_FOLDER: Path = Path(__file__).parent

# the maximum number of validation instances
MAX_VALIDATION_SIZE: int = 5000


if __name__ == "__main__":
    parser = HfArgumentParser([TrainingLogicDatasetArguments, TrainingArguments])
    datapipeline_args: TrainingLogicDatasetArguments = parser.parse_args_into_dataclasses()[0]
    training_args: TrainingArguments = parser.parse_args_into_dataclasses()[1]

    if datapipeline_args.model_type == "dec":
        model = AutoModelForCausalLM.from_pretrained(
            datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length
        )
    elif datapipeline_args.model_type == "enc-dec":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            datapipeline_args.model_name, max_length=datapipeline_args.max_gen_length
        )

    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.tokenizer_dir, padding_side="right")

    model.resize_token_embeddings(len(tokenizer))

    train_dataset_path = Path(datapipeline_args.tokenized_data)
    if train_dataset_path.is_dir():
        dataset = load_dataset(datapipeline_args.train_data, split="train", num_proc=os.cpu_count())
    else:
        dataset = load_dataset(
            train_dataset_path.parent.as_posix(),
            data_files=train_dataset_path.parts[-1],
            split="train",
            num_proc=os.cpu_count(),
        )

    # select columns that matter for train
    dataset = dataset.select_columns(["input_ids", "labels"])

    # use the train/test split to create the validation set
    # compute test size
    if datapipeline_args.validation_frac:
        validation_size = int(len(dataset) * datapipeline_args.validation_frac)
        validation_size = min(MAX_VALIDATION_SIZE, validation_size)

        dataset = dataset.train_test_split(test_size=validation_size, seed=training_args.seed)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    aim_callback = AimCallback(experiment="logic_flow_train")
    aim_callback.experiment["dvc_info"] = DvcData(ROOT_FOLDER)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=datapipeline_args.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if datapipeline_args.validation_frac else dataset,
        eval_dataset=dataset["test"] if datapipeline_args.validation_frac else None,
        data_collator=data_collator,
        callbacks=[aim_callback, early_stopping_callback],
    )

    result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_state()
