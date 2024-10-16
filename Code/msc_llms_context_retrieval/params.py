"""Define the script parameters for the logic pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#from outsystems.ai.aws.s3 import RD_AI_TEAM_BUCKET

#from outsystems.ai.language_model.transformations.logic.text.enums import (
#    RepresentationType,
#)


def parse_none_or_int(value: Optional[str | int]) -> Optional[int]:
    """Parse none or integer input.

    Args:
        value (Optional[str | int]): Value to be parsed.

    Returns:
        Optional[int]: Processed value.
    """
    if value is None:
        return value
    elif value == "None":
        return None
    else:
        return int(value)


def parse_none_or_str(value: Optional[str]) -> Optional[str]:
    """Parse none or string input.

    Args:
        value (Optional[str]): Value to be parsed.

    Returns:
        Optional[str]: Processed value.
    """
    if value is None:
        return value
    elif value == "None":
        return None
    else:
        return value


@dataclass
class DatasetArguments:
    """Arguments related to the dataset."""

    dataset_name: Optional[str] = field(
        default="XRayEncoding-20240103", metadata={"help": "Xray dataset to be used in the extraction."}
    )
    subset: Optional[str] = field(
        default=None, metadata={"help": "Subset of files to be used. To use all subsets use None."}
    )
    override_activation_code: Optional[bool] = field(
        default=True, metadata={"help": "Get the activation code from the filepath."}
    )

    setup: Optional[str] = field(
        default="logic", metadata={"help": "Dataset default setup. Either logic or data_model."}
    )

    def __post_init__(self) -> None:
        """Run verifications after init."""
        # when no subset is given in the config we need to parse the string None
        self.subset = parse_none_or_int(self.subset)


@dataclass
class AggregateDataClassModulesArguments:
    """Arguments related to the aggregation of the Data Class Modules."""

    applications_dataset_name: str = field(
        default="20230327-ListApplications-20230327-17-07.jsonlines.gz",
        metadata={"help": "List Applications dataset to be used in the aggregation."},
    )


@dataclass
class FilterDataClassDatasetArguments:
    """Arguments that control the type of tokenizer used."""

    tokenizer_name: str = field(default="Salesforce/codegen-350M-mono", metadata={"help": "Tokenizer name"})
    attribute_threshold: float = field(default=0.15, metadata={"help": "Threshold value for attribute filter"})


@dataclass
class TextDataModelDatasetArguments:
    """Arguments that control the text generation."""

    include_records: bool = field(
        default=False, metadata={"help": "When True it also encodes static entity records into text "}
    )


@dataclass
class TextualLogicDatasetArguments:
    """Arguments that control the conversion to textual representation."""

    noise_add_probability: float = field(
        metadata={"help": "probability of considering an entity or structure that is not used."}
    )
    max_noise_entities_or_structures: float = field(
        metadata={"help": "maximum percentage of unused entities and or structrues that can be considered"}
    )
    representation_type: str = field(default="python", metadata={"help": "Representation type to convert the flows."})
    use_parameters: bool = field(
        default=False, metadata={"help": "When True it also encodes the node parameters. Default False."}
    )


@dataclass
class RepresentationTypeArgument:
    """Arguments that control the representation type argument."""

    representation_type: str = field(default="python", metadata={"help": "Representation type to convert the flows."})


@dataclass
class SplitArguments:
    """Arguments related to the dataset splitting."""

    test_size: float = field(default=0.1, metadata={"help": "Percentage of the factories to keep in the test set."})
    setup: Optional[str] = field(
        default="logic", metadata={"help": "Dataset default setup. Either logic or data_model."}
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Flag to indicate if the keys should (or not) be shuffled before splitting."},
    )
    random_seed: int = field(
        default=0,
        metadata={"help": "Random seed for the shuffling. Only used if `shuffle` is set to `True`."},
    )


@dataclass
class ModelArguments:
    """Arguments related to the model."""

    model_type: str = field(metadata={"help": "Defines the model type. I can be enc-dec or dec."})
    tokenizer_name: str = field(metadata={"help": "Tokenizer name to be used"})
    model_name: Optional[str] = field(metadata={"help": "Model name to be used"})
    max_number_of_train_tokens: int = field(metadata={"help": "Maximum number of tokens of the train instances."})


@dataclass
class SubflowsArguments:
    """Arguments for the splitting into sub-flows."""

    split_into_subflows: bool = field(default=False, metadata={"help": "If true, generate sub-flows"})
    max_num_sub_flows: int = field(default=10, metadata={"help": "Maximum number of sub-flows to generate per flow."})
    include_complete_flow: bool = field(default=False, metadata={"help": "If True, the complete flow is included."})
    min_number_of_nodes: int = field(default=2, metadata={"help": "Minimum number of nodes of the sub-flows."})
    max_number_of_nodes: int = field(default=40, metadata={"help": "Maximum number of nodes of the sub-flows."})


@dataclass
class MaskingArguments:
    """Arguments for the masking of the flows."""

    max_num_instances: int = field(default=10, metadata={"help": "Maximum number of nodes to mask, per flow."})


@dataclass
class ContextObfuscationArguments:
    """Arguments for the context obfuscation."""

    graph_obfuscation_prob: float = field(metadata={"help": "probability of obfuscating a graph."})
    n_obfuscations: int = field(metadata={"help": "number of obfuscated graphs to generate."})
    graph_prop_default_obfuscation_prob: float = field(
        metadata={"help": "graph property obfuscation probability. Must be between 0 and 1."}
    )
    node_prop_default_obfuscation_prob: float = field(
        metadata={"help": "default obfuscation probability used for all properties"}
    )
    cascade_delete_node_properties: dict[str, dict[str, set[str]]] = field(
        default=None, metadata={"help": "mapping to enable cascade deletion of properties"}
    )
    node_prop_obfuscation_prob: dict[str, dict[str, float]] = field(
        default=None, metadata={"help": "obfuscation probabilities for specific properties"}
    )


@dataclass
class TestDatasetArguments(DatasetArguments, SubflowsArguments, TextualLogicDatasetArguments):
    """Arguments related to the test dataset."""

    dataset_partition: str = field(default="train", metadata={"help": "Dataset partition: train/test."})
    max_num_instances: int = field(default=10, metadata={"help": "Maximum number of nodes to mask, per flow."})
    random_seed: int = field(default=42, metadata={"help": "The random seed to use."})
    max_instances: int = field(default=10000, metadata={"help": "The maximum number of instances."})


@dataclass
class TrainingLogicDatasetArguments(ModelArguments):
    """Arguments related to the training."""

    tokenizer_dir: str = field(metadata={"help": "Defines the path where the tokenizer will be"})
    training_type: str = field(metadata={"help": "Defines the training type. I can be causal, masking or ul2."})
    tokenized_data: str = field(metadata={"help": "path to the parquet with the training tokenized data"})
    train_data: str = field(default="", metadata={"help": "path to the parquet with the training data"})
    early_stopping_patience: int = field(default=2, metadata={"help": "early stopping patience."})
    max_gen_length: Optional[int] = field(default=512, metadata={"help": "The max number of tokens to generate."})
    num_cpus: Optional[int] = field(
        default=os.cpu_count(),
        metadata={"help": "Number of cpus to run the data processing. Default value equals to the number of cores."},
    )
    validation_frac: Optional[float] = field(
        default=0.3, metadata={"help": "Fraction of training dataset to be used for validation."}
    )
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Maximum input length."})
    masking_seed: Optional[int] = field(default=42, metadata={"help": "Seed to compute the masks."})
    truncation: Optional[bool] = field(default=False, metadata={"help": "When True it truncates the input."})


@dataclass
class TestLogicArguments:
    """Arguments related to testing the model."""

    model_dir: str = field(metadata={"help": "Defines the path to a previously fine-tuned model checkpoint."})
    verbose: int = field(default=1, metadata={"help": "Verbosity level."})
    batch_size: int = field(default=64, metadata={"help": "Batch size used for inference."})
    num_generate_sequences: int = field(default=5, metadata={"help": "Number of sequences to generate."})
    random_seed: int = field(default=42, metadata={"help": "The random seed to use."})
    max_instances: int = field(default=10000, metadata={"help": "The maximum number of instances."})
    max_number_of_test_tokens: int = field(default=2048, metadata={"help": "The maximum number of test tokens."})
    max_new_tokens: int = field(default=256, metadata={"help": "The maximum number of generate tokens."})
    representation_type: str = field(default="python", metadata={"help": "Representation type to convert the flows."})


@dataclass
class SampleTestArguments:
    """Arguments related to test set sampling."""

    sample_size: int = field(metadata={"help": "Size of the sample set."})




@dataclass
class DataModelCodeLlama2PythonTrainingArguments:
    """Arguments related to Code Llama 2 - Python training for datamodel suggestions."""

    lora_r: int = field(default=16, metadata={"help": "Low rank of LoRA matrices to be learned during finetuning."})
    lora_alpha: int = field(default=64, metadata={"help": "Alpha parameter for LoRA scaling."})
    lora_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for LoRA layers."})
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length used for training."})
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility."})
    gradient_accumulation_steps: int = field(default=64, metadata={"help": "Number of gradient accumulation steps."})


@dataclass
class DataModelDeepSeekCoderTrainingArguments:
    """Arguments related to DeepSeek Coder training for datamodel suggestions."""

    use_weighted_loss: bool = field(
        default=False, metadata={"help": "When True it computes weighted loss based on instances key tokens."}
    )
