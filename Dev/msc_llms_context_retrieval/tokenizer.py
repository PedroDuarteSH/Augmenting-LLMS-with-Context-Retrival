import os
import subprocess

def tokenize_dataset(training_path, tokenized_data_dir, parameters):
    # Construct the command as a list of arguments
    command = [
        "python",
        "tokenize_codet5.py",
        "--train_data",
        os.path.join(training_path),
        *parameters.split(),  # Split parameters into a list
        "--tokenized_data",
        os.path.join(tokenized_data_dir),
    ]

    # Join the command list into a single string for printing
    command_str = " ".join(command)

    # Execute the command using subprocess.run
    try:
        subprocess.run(command, check=True, shell=False)
        print(f"Command executed successfully: {command_str}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command_str}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output.decode('utf-8')}")

if __name__ == "__main__":
    currentPath = os.getcwd()
    #parameters = """--model_name Salesforce/codet5-small --tokenizer_name Salesforce/codet5-small --tokenizer_dir tokenizer --max_gen_length 256 --model_type enc-dec --training_type masking --max_number_of_train_tokens 2048"""
    parameters = """--model_name nomic-ai/nomic-bert-2048 --tokenizer_name nomic-ai/nomic-bert-2048 --tokenizer_dir encoder_tokenizer --model_type enc --training_type masking --max_number_of_train_tokens 2048"""
    training_file = os.path.join(currentPath, "msc_llms_context_retrieval/datasets/logic/train_python_v1_dataset.parquet")
    tokenizer_file = os.path.join(training_file.split('.')[0]+ '_tokenized.parquet')
    
    if os.path.isdir(training_file):
        if not os.path.exists(tokenizer_file):
            os.mkdir(tokenizer_file)
        tokenized_files = os.listdir(tokenizer_file)

        for i, file in enumerate(os.listdir(training_file)):
            if(file not in tokenized_files):
                print(f"Tokenizing {file} ({i})")
                tokenize_dataset(os.path.join(training_file, file), os.path.join(tokenizer_file, file), parameters)
            else:
                print(f"File {file} ({i}) already tokenized")
    else:
        print(f"Tokenizing {training_file}")
        tokenize_dataset(training_file, tokenizer_file, parameters)

    