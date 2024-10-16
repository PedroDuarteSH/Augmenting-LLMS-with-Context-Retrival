from dataclasses import dataclass, field
import logging
from datetime import datetime
import os
from pathlib import Path

@dataclass
class LoggerArguments:
    log_file: str = field(
        metadata={"help": "Path to the log file."},
        default="./log/test_results_encoder.log"
    )
    log: bool = field(
        metadata={"help": "Log the results."},
        default=True
    )
    
    def get_log_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
    
    def setup_logging(self, model_name):
        log_path = Path(self.log_file)
        if not log_path.parent.exists():
            os.makedirs(log_path.parent, exist_ok=True)
        if log_path.exists():
            print(f"Log file {log_path} already exists. Overwriting...")
            with open(log_path, 'w') as f:
                f.write('')
        logging.basicConfig(filename= self.log_file, encoding='utf-8', level=logging.DEBUG)
        logging.debug(self.get_log_time() + f"Starting inference of the {model_name} model")
    
    def log_message(self, message):
        if self.log:
            logging.debug(self.get_log_time() + message)
            print(self.get_log_time() + message)
        else:
            print(message)

@dataclass
class TestArguments:
    model_name: str = field(
        metadata={"help": "Path to the model."},
        default="./model/logic_flows_finetune_v2_node_context_extended/model"
    )
    model_properties_name: str = field(
        metadata={"help": "Path to the model."},
        default="./model/logic_flows_finetune_v2_properties/model"
    )
    max_gen_length: int = field(
        metadata={"help": "Maximum length of the generated sequence."},
        default=256
    )
    test_file: str = field(
        metadata={"help": "Path to the test file."},
        default="./msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
    )
    output_file: str = field(
        metadata={"help": "Path to the output file. If it is a directory, the output files will be saved in the directory with this name."},
        default="./msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_properties_extended_results.parquet"
    )
    num_return_sequences: int = field(
        metadata={"help": "Number of sequences to generate."},
        default=5
    )
    num_beams: int = field(
        metadata={"help": "Number of beams to use."},
        default=10
    )
    device: str = field(
        metadata={"help": "Device to use."},
        default="cuda"
    )

@dataclass
class ExtractDatasetArguments:
    dataset_path: str = field(
        metadata={"help":"File/Folder path to be processed"},
        default="./msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
    )
    output_path: str = field(
        metadata={"help":"Output Path to the processed. 3 Files (\"flow_only\", \"flow_dataset\" and \"flow_imports\") will be created, this is base path."},
        default="./msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
    )

@dataclass
class EncoderDatasetGeneratorArguments:
    full_dataset_path: str = field(
        metadata={"help":"File/Folder path to be processed"},
        default="msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
    )
    flow_only_dataset_path: str = field(
        metadata={"help":"Dataset with only the flow context to have its labels replaced to the encoder form"},
        default="msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset.parquet"
    )
    output_dataset_path: str = field(
        metadata={"help":"Output Path to the processed dataset"},
        default="msc_llms_context_retrieval/datasets/logic/test_python_v1_dataset_encoder_reduced.parquet"
    )


@dataclass
class ProcessDataset:
    dataset_path: str = field(
        metadata={"help":"File/Folder path to be processed"},
        default="msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only.parquet"
    )
    output_path: str = field(
        metadata={"help":"Output Path to the processed dataset"},
        default="msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset_flow_only_reduced.parquet"
    )