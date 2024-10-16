from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer
from params import TrainingLogicDatasetArguments
from pathlib import Path
from datasets import load_dataset
import os
from tokenizers import AddedToken
from transformers.tokenization_utils_base import TruncationStrategy
from typing import Any
import pandas as pd
# literal token, to replace hashed texts
LITERAL_TOKEN = "<LITERAL>"

# end token
END_TOKEN: str = "<END>"

SEP_TOKEN = "<s>"
MASK_TOKEN = "[MASK]"

NODE_TOKENS = [
    "IIfNode",
    "IJSONSerializeNode",
    "INRSendEmailNode",
    "IAssignment",
    "ISendEmailNode",
    "IRaiseExceptionNode",
    "IRecordListToExcelNode",
    "IForEachNode",
    "IExcelToRecordListNode",
    "ISQLNode",
    "IAggregateNode",
    "<END>",
    "IJSONDeserializeNode",
    "IExecuteServerActionNode"
]

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

 


    # assert there are no unknown ids
    #assert not any(tokenizer.unk_token_id in tokenized_text for tokenized_text in text_tokenization["input_ids"])
    #assert not any(tokenizer.unk_token_id in tokenized_label for tokenized_label in label_tokenization["input_ids"])

    return {"input_ids": text_tokenization["input_ids"], "num_tokens": len(text_tokenization["input_ids"])}

if __name__ == "__main__":
    parser = HfArgumentParser([TrainingLogicDatasetArguments])
    datapipeline_args: TrainingLogicDatasetArguments = parser.parse_args_into_dataclasses()[0]
    
    tokenizer = AutoTokenizer.from_pretrained(datapipeline_args.tokenizer_name, padding_side="right")

    
    # add <LITERAL> and <END> token
    tokenizer.add_tokens([LITERAL_TOKEN, END_TOKEN], special_tokens=False)
    
    # add node tokens
    tokenizer.add_tokens(NODE_TOKENS, special_tokens=False)

    print(tokenizer.mask_token)
    if not tokenizer.mask_token and datapipeline_args.training_type == "masking":
        tokenizer.add_tokens([MASK_TOKEN], special_tokens=True)
        tokenizer.mask_token = MASK_TOKEN

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

    dataset = dataset.map(
        lambda x: {"text": x["text"].replace("<extra_id_0>", 
                                               MASK_TOKEN)},
    )

    # tokenize input text and target labels
    dataset = dataset.map(
        tokenize_batch,
        fn_kwargs={"tokenizer": tokenizer, "add_special_tokens": True, "truncation": False},
        batched=True,
        num_proc=16,
    )

    total_instances = len(dataset)

     # filter xre invalid expressions
    dataset = dataset.filter(
        lambda x: not (
            XRE_INVALID_REF_OBJ in (x["text"])
            or XRE_UNK_REF_OBJ in (x["text"])
            or XRE_INVALID_EXP in (x["text"])
        )
    )
    print(f"Discarded (xre): {total_instances-len(dataset)}/{total_instances}")

    # save tokenizer
    tokenizer.save_pretrained(datapipeline_args.tokenizer_dir)

    # save dataset
    dataset.to_parquet(datapipeline_args.tokenized_data)
