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

parking_data = pd.read_hdf('nottingham.h5')
true = np.load('ytrue.npy')
pred = np.load('ypred.npy')
instruction_text = 'Role: You are an AI agent responsible for predicting the availability of parking lots. Objective: Your task is to forecast the number of available parking lots for the next 2 hours. To do this, you will analyze data from the past 2 hours of parking records, along with simulation-based predictions for the next 2 hours and other additional information that could affect the parking lots availability prediction values. Input Data: (1) Car park description: {}. (2) Prediction horizon: you are given 2 hours historical input from {} to {} on {}, and you are supposed to predict future parking lots from {} to {}. (3) The Natural patterns of parking availability (e.g., peak and off-peak periods throughout the day) are as follow: {}. (4) Historical Records: Parking data for the past 2 hours {}. (5) Simulation predictions: Forecasts for parking availability for the next 2 hours based on simulation models {}. Consider all provided information (historical records, simulation predictions, and additional factors) to predict parking lot availability for the next 2 hours with accuracy. Please analyze the data and provide only your final numerical predictions. DO NOT include any additional content in your answer. Strictly enclose your answer inside square brackets [] and provide exactly 8 numerical values.'
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

with open("answer_list.json", "r") as file:
    # loaded_list = json.load(file)
    content = file.read()

    if not content.strip():
        raise ValueError("File is empty")

    data = json.loads(content)
loaded_list=data
import re
match_list = []
numerical_list = []
for i in range(len(loaded_list)):
    # Input string
    input_string = loaded_list[i]

    # Regular expression to capture values inside [] after [/INST]
    pattern = r'\[/INST\]\s*\[([^\]]+)\]'
    # pattern = r'\[/INST\]\s*\[([0-9.\s]+)\]\s*</s>'

    # Perform the search and extract matches
    matches = re.findall(pattern, input_string)
    # if len(matches) == 0:
    #     match_list.append(test_pred[i])
    # else:
    #     match_list.append(matches)
    if matches:
        numerical_values = matches[0].split()
        numerical_list.append(numerical_values)
        # print(numerical_values)
        # numerical_array = np.array([float(num) for num in numerical_values])
        # print(type(numerical_values))
    else:
        numerical_list.append(list(test_pred[i]))

LLM_pred = np.zeros((19*2163, 8))
for i in range(len(numerical_list)):
    try:
        if type(numerical_list[i][-1]) == str:    
            numerical_values=numerical_list[i]
            re_list = []
            if len(numerical_values) < 8:
                numerical_values.append(numerical_values[-1])
            for j in range(8):
                # print(i,j)
                if numerical_values[j] == '-.':
                    numerical_values[j] = numerical_values[j-1]
                if numerical_values[j][-1] != '.':
                    re_list.append(numerical_values[j])
                else:
                    re_list.append(numerical_values[j][:-1])
            numerical_array = np.array([float(num) for num in re_list])
            LLM_pred[i,:]=numerical_array
        else:
            LLM_pred[i,:]=numerical_list[i]
    except ValueError:
        LLM_pred[i, :] = test_pred[i]  # Assign test_pred[i] if ValueError occurs
        print(f"ValueError encountered at index {i}, using test_pred[i] instead.")
    except IndexError:
        LLM_pred[i, :] = test_pred[i]  # Assign test_pred[i] if ValueError occurs
        print(f"IndexError encountered at index {i}, using test_pred[i] instead.")
# LLM_pred

LLM_p = np.reshape(LLM_pred, (2163,19,8), order='F')


test_pred = np.reshape(test_pred, (2163,19,8), order='F')
test_truth = np.reshape(test_truth, (2163,19,8), order='F')

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = labels!=null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # print('preds shape is: ', preds.shape)
    # print('labels shape is: ', labels.shape)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        # update if null_val is 0
        mask = ((labels-null_val).abs() > 1e-5)
        # mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

LLM_p = torch.tensor(LLM_p)

# LLM_p = torch.tensor(LLM_pred)
test_truth =torch.tensor(test_truth)
test_pred =torch.tensor(test_pred)

maes = []
rmses = []
mapes = []
for i in range(8): 
    mae = masked_mae(LLM_p[:,:,i], test_truth[:,:,i],0.0)
    mse = masked_rmse(LLM_p[:,:,i], test_truth[:,:,i],0.0)
    mape = masked_mape(LLM_p[:,:,i], test_truth[:,:,i],0.0)
    maes.append(mae.item())
    rmses.append(mse)
    mapes.append(mape)
print(maes)
print(rmses)
print(mapes)

maes = []
rmses = []
mapes = []
for i in range(8): 
    mae = masked_mae(test_pred[:,:,i], test_truth[:,:,i],0.0)
    mse = masked_rmse(test_pred[:,:,i], test_truth[:,:,i],0.0)
    mape = masked_mape(test_pred[:,:,i], test_truth[:,:,i],0.0)
    maes.append(mae.item())
    rmses.append(mse)
    mapes.append(mape)
print(maes)
print(rmses)
print(mapes)
