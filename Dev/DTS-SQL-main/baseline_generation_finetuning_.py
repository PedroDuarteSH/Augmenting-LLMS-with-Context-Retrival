import os
import torch
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset
from sql_metadata import Parser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import gc
def formatting_prompts_func(training_dataset):
  output_texts = []
  for i in range(len(training_dataset['question'])):
    question = training_dataset['question'][i]
    database_schema = training_dataset['database_schema'][i]
    query = training_dataset['query'][i]
    user_message = f"""Given the following SQL tables, your job is to generate the Sqlite SQL query given the user's question.
Put your answer inside the ```sql and ``` tags.
{database_schema}
###
Question: {question}
"""
    assitant_message = f"""
```sql
{query} ;
```
<|EOT|>
"""
    messages = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assitant_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    output_texts.append(text)
  return output_texts
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_has_fp16_weight = True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant = True
    )
    flush()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #attn_implementation="flash_attention_2",
        torch_dtype = torch.float16,
        device_map='auto',
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    print(model)
    
    data_files = {"train": "train/filtered_finetuning_dataset.csv", "validation": "validation/filtered_validation_dataset.csv"}
    dataset = load_dataset('csv', data_files=data_files)
    print(dataset)
    
    response_template = "### Response:" #deepseek
    # response_template = "[/INST]" #mistral"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    
    # Training Config
    lora_r = 64
    lora_alpha = 32
    lora_dropout = 0.1
    output_dir = "./SFT_baseline_without_steps"
    num_train_epochs = 3
    bf16 = False
    fp16 = True
    overwrite_output_dir = True
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 8
    gradient_checkpointing = True
    evaluation_strategy = "steps"
    learning_rate = 5e-5
    weight_decay = 0.01
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.01
    max_grad_norm = 0.3
    group_by_length = True
    auto_find_batch_size = False
    save_steps = 50
    logging_steps = 50
    load_best_model_at_end= False
    packing = False
    save_total_limit=3
    neftune_noise_alpha=5
    #report_to="wandb"
    max_seq_length = 1000#set based on the maximum number of tokens
    
    
    #LORA CONFIG
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Training Arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=load_best_model_at_end,
        per_device_train_batch_size=per_device_train_batch_size,
        evaluation_strategy=evaluation_strategy,
        max_grad_norm = max_grad_norm,
        auto_find_batch_size = auto_find_batch_size,
        save_total_limit = save_total_limit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bf16=bf16,
        fp16=fp16,
        
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=None,
        neftune_noise_alpha= neftune_noise_alpha
    )
    
   
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=max_seq_length,
        packing=packing
    )
    
    os.environ["WANDB_MODE"] = "disabled"


    trainer.train()
    
    output_dir = os.path.join("./", "final_checkpoint_baseline")
    trainer.model.save_pretrained(output_dir)
    