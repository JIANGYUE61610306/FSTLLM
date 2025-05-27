# FSTLLM
This is the implementation for FSTLLM: Spatio-Temporal LLM for Few-Shot Time Series Forecasting.
# Requirements
```bash
pip install -r requirements.txt
```

# Step 1
Please run the STGNN backbone to obtain numerical prediction tokens

```bash
python train.py --config_filename=data/model/para_not.yaml
```
# Step 2
Please fine-tune FSTLLM with
```bash
FSTLLM_fine_tune.py
```
# Step 3
Please evaluate FSTLLM with
```bash
FSTLLM_inference.py
```
# Step 4
Please print evaluation results with
```bash
FSTLLM_evaluation.py
```
