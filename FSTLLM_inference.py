import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import json



file_path = "generated_dataset.txt"

# Initialize an empty list to store the loaded data
loaded_list = []

# Open the file in read mode
with open(file_path, "r") as file:
    # Read each line of the file
    for line in file:
        # Strip whitespace from the line
        line = line.strip()
        # Check if the line is not empty
        if line:
            try:
                # Convert the line to an integer and append it to the list
                loaded_list.append(line)
            except ValueError:
                # Handle the case when the line cannot be converted to an integer
                print("Warning: Skipped line as it cannot be converted to an integer:", line)

# print("Loaded list:", loaded_list)
len(loaded_list)

random.shuffle(loaded_list)


from datasets import Dataset, DatasetDict
# my_dict = {'text':['1', '2', '3']}
# my_dict = {'text':dataset1_one_ans[:1000]}
my_dict = {'text':loaded_list}
dataset1 = Dataset.from_dict(my_dict)

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-nottingham"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load dataset (you can process it here)
# dataset = load_dataset(dataset_name, split="train")
dataset = dataset1
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
# trainer.train()

# Save trained model
# trainer.model.save_pretrained(new_model)


pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1500)

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
# new_model = "llama-2-metr-2"
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1500)
instruction_text = 'Role: You are an AI agent responsible for predicting the availability of parking lots. Objective: Your task is to forecast the number of available parking lots for the next 2 hours. To do this, you will analyze data from the past 2 hours of parking records, along with simulation-based predictions for the next 2 hours and other additional information that could affect the parking lots availability prediction values. Input Data: (1) Car park description: {}. (2) Prediction horizon: you are given 2 hours historical input from {} to {} on {}, and you are supposed to predict future parking lots from {} to {}. (3) The Natural patterns of parking availability (e.g., peak and off-peak periods throughout the day) are as follow: {}. (4) Historical Records: Parking data for the past 2 hours {}. (5) Simulation predictions: Forecasts for parking availability for the next 2 hours based on simulation models {}. Consider all provided information (historical records, simulation predictions, and additional factors) to predict parking lot availability for the next 2 hours with accuracy. Please analyze the data and provide only your final numerical predictions. DO NOT include any additional content in your answer. Strictly enclose your answer inside square brackets [] and provide exactly 8 numerical values.'
parking_data = pd.read_hdf('nottingham.h5') ### address refer to FSTLLM_STGNN/data/
true = np.load('ytrue.npz') ### address refer to FSTLLM_STGNN/data/
pred = np.load('ypred.npz') ### address refer to FSTLLM_STGNN/data/
carpark_des_list = []
file_path = "carpark_des_list.txt"
# Open the file in read mode
with open(file_path, "r") as file:
    # Read each line of the file
    for line in file:
        # Strip whitespace from the line
        line = line.strip()
        # Check if the line is not empty
        if line:
            try:
                # Convert the line to an integer and append it to the list
                carpark_des_list.append(line)
            except ValueError:
                # Handle the case when the line cannot be converted to an integer
                print("Warning: Skipped line as it cannot be converted to an integer:", line)

file_path = "natural_pattern_list.txt"

# # Initialize an empty list to store the loaded data
natural_pattern_list = []

# Open the file in read mode
with open(file_path, "r") as file:
    # Read each line of the file
    for line in file:
        # Strip whitespace from the line
        line = line.strip()
        # Check if the line is not empty
        if line:
            try:
                # Convert the line to an integer and append it to the list
                natural_pattern_list.append(line)
            except ValueError:
                # Handle the case when the line cannot be converted to an integer
                print("Warning: Skipped line as it cannot be converted to an integer:", line)

generated_dataset = []
test_truth = []
test_pred = []
for j in range(19):
    for i in range(2163): ###2163 is the testset num 
        t1 = parking_data.iloc[8658-8+i:8658+i,0].index[0]
        t2 = parking_data.iloc[8658-8+i:8658+i,0].index[-1]
        t3 = parking_data.iloc[8658+i:8658+8+i,0].index[0]
        t4 = parking_data.iloc[8658+i:8658+8+i,0].index[-1]
        weekday = parking_data.iloc[8658+i:8658+8+i,0].index[0].day_name()
        Historical = parking_data.iloc[8658-8+i:8658+i,j].values
        prediction = pred[i,j,:]
        truth = true[i,j,:]
        carpark_des = carpark_des_list[j]
        natural_pattern = natural_pattern_list[j]
        sample_prompt = instruction_text.format(carpark_des,t1, t2, weekday, t3, t4, natural_pattern, Historical, prediction)
        generated_dataset.append(sample_prompt)
        test_truth.append(truth)
        test_pred.append(prediction)
loaded_list=generated_dataset




answer_list = []
for i in tqdm(range(len(loaded_list))):
    prompt = loaded_list[i]
    result = pipe(f"<s>[INST] {prompt} [/INST]")

    string_data = result[0]['generated_text']
    answer_list.append(string_data)


# Open a file in write mode
with open("answer_list.json", "w") as file:
    # Save the list as a JSON array
    json.dump(answer_list, file)
