"""Tokenization script."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

from params import TrainingLogicDatasetArguments

# literal token, to replace hashed texts
LITERAL_TOKEN = "<LITERAL>"

# end token
END_TOKEN: str = "<END>"

SEP_TOKEN = "<s>"
MASK_TOKEN = "<extra_id_0>"

# TODO - mv to x-ray constants
# x-ray expressions with unknown referred objects
XRE_UNK_REF_OBJ: str = "#unknown#"
# x-ray invalid expressions
XRE_INVALID_EXP: str = "#invalid#"
XRE_INVALID_REF_OBJ: str = "!Invalid!"


def tokenize_batch(
    batch: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = True,
    truncation: bool | str | TruncationStrategy = False,
) -> dict[str, Any]:
    """Tokenize the code.

    Args:
        batch (pd.DataFrame): Batch to be tokenized.
        tokenizer (PreTrainedTokenizer): Tokenizer to be used.
        add_special_tokens (bool, optional): whether or not to encode the sequences with the special
                                             tokens relative to their model. Defaults to True.
        truncation (bool | str | TruncationStrategy, optional): activates and controls truncation. Defaults to False.

    Returns:
        pd.DataFrame: Tokenized batch.
    """
    # compute the input token ids
    text_tokenization = tokenizer(batch["text"], add_special_tokens=add_special_tokens, truncation=truncation)

    # if there is a label, compute the label token ids too, useful for training
    label_tokenization = tokenizer(
        # replace any possible `None` by `""`
        # the `extra_id_0` and `extra_id_1` tokens are form the `T5Tokenizer`.
        [f"<extra_id_0>{x}<extra_id_1>" or "" for x in batch["label"]],
        add_special_tokens=add_special_tokens,
        truncation=truncation,
    )

    # assert there are no unknown ids
    assert not any(tokenizer.unk_token_id in tokenized_text for tokenized_text in text_tokenization["input_ids"])
    assert not any(tokenizer.unk_token_id in tokenized_label for tokenized_label in label_tokenization["input_ids"])

    return {"input_ids": text_tokenization["input_ids"], "labels": label_tokenization["input_ids"]}



if __name__ == "__main__":
    parser = HfArgumentParser([TrainingLogicDatasetArguments])
    datapipeline_args: TrainingLogicDatasetArguments = parser.parse_args_into_dataclasses()[0]
    
    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.tokenizer_name, padding_side="right")

    # hotfix: if using codet5, the tokenizer is left-stripping the special tokens.
    # The direct impact of this is that it removes indentation and new lines.
    if datapipeline_args.model_name == "Salesforce/codet5-small":
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(at.content, rstrip=False, lstrip=False, single_word=False, normalized=True)
                    for at in tokenizer.special_tokens_map_extended["additional_special_tokens"]
                ]
            },
            replace_additional_special_tokens=True,
        )

    # add <LITERAL> and <END> token
    tokenizer.add_tokens([LITERAL_TOKEN, END_TOKEN], special_tokens=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not tokenizer.mask_token and datapipeline_args.training_type == "masking":
        tokenizer.add_tokens([MASK_TOKEN], special_tokens=True)
        tokenizer.mask_token = MASK_TOKEN

    if not tokenizer.sep_token and datapipeline_args.model_type == "dec":
        tokenizer.add_tokens([SEP_TOKEN], special_tokens=True)
        tokenizer.sep_token = SEP_TOKEN

    train_dataset_path = Path(datapipeline_args.train_data)
    
    if train_dataset_path.is_dir():
        print("Loading dataset from local dir")
        dataset = load_dataset(datapipeline_args.train_data, split="train", num_proc=os.cpu_count())
    else:
        print("Loading dataset from HF")
        dataset = load_dataset(
            "parquet",
            data_files={"train": datapipeline_args.train_data},
            split="train",
            num_proc=os.cpu_count(),
        )

    # tokenize input text and target labels
    dataset = dataset.map(
        tokenize_batch,
        fn_kwargs={"tokenizer": tokenizer, "add_special_tokens": True, "truncation": False},
        batched=True,
        num_proc=os.cpu_count(),
    )

    total_instances = len(dataset)

    # filter instances that do not fit in the total number of tokens we are enabling
    if datapipeline_args.max_number_of_train_tokens is not None:
        print(f"Filtering instances with more than {datapipeline_args.max_number_of_train_tokens} tokens")
        dataset = dataset.filter(
            lambda x, max_number_of_train_tokens = datapipeline_args.max_number_of_train_tokens, model_max_length = datapipeline_args.model_max_length: (
                len(x["input_ids"]) <= max_number_of_train_tokens
                and len(x["labels"]) <= model_max_length
            ),
            num_proc=os.cpu_count(),
        )

        print(f"Discarded (#tokens): {total_instances-len(dataset)}/{total_instances}")
        total_instances = len(dataset)

    # filter xre invalid expressions
    dataset = dataset.filter(
        lambda x: not (
            XRE_INVALID_REF_OBJ in (x["text"] + " " + x["label"])
            or XRE_UNK_REF_OBJ in (x["text"] + " " + x["label"])
            or XRE_INVALID_EXP in (x["text"] + " " + x["label"])
        )
    )
    print(f"Discarded (xre): {total_instances-len(dataset)}/{total_instances}")

    # save tokenizer
    tokenizer.save_pretrained(datapipeline_args.tokenizer_dir)

    # save dataset
    dataset.to_parquet(datapipeline_args.tokenized_data)
