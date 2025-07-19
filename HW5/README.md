# Homework 5: Fine-tuning is Powerful

This repository contains the implementation for ML Spring 2025 Homework 5, focusing on fine-tuning large language models using LoRA (Low-Rank Adaptation) techniques.

## Overview

This project demonstrates the power of fine-tuning by adapting a pre-trained Llama-2-7b model using the Unsloth framework. The goal is to improve the model's performance on conversational tasks through efficient parameter-efficient fine-tuning.

## Features

### 1. Model Configuration

- **Base Model**: `unsloth/Llama-2-7b-bnb-4bit` (4-bit quantized)
- **LoRA Parameters**: Configurable rank (r), alpha, and target modules
- **Memory Optimization**: 4-bit quantization and gradient checkpointing

### 2. Dataset Processing

- **Training Data**: FastChat Alpaca 52k conversations
- **Test Data**: Evol-Instruct 150 examples
- **Data Filtering**: Removes empty conversations and adds text fields
- **Sorting Options**: 
  - The dataset plot shows the conversion and corresponding scores, with a notable hotspot observed between lengths 200 and 300, where scores range from 3 to 5.

### 3. Training Features

- **Parameter-Efficient Fine-tuning**: Uses LoRA adapters
- **Curriculum Learning**: Optional progressive training from simple to complex examples
- **Chat Template**: Llama-3.1 template for better performance
- **Training on Responses Only**: Focuses loss computation on assistant outputs

## Usage

### 1. Training the Model

Open the Jupyter notebook `Homework5_Finetuning_is_Powerful.ipynb` and run the cells sequentially:

1. **Check Environment and Package**: After pip install check torch version and cuda version. And check import successfully.
2. **Execution**: Run the code.

### 2. Evaluation

The evaluation process uses OpenAI's API to assess the quality of model responses through automated scoring.

#### Setup & Execution
1. **Configure API**: Add your OpenAI API key to `evaluate.py`
2. **Run Evaluation**: Execute the evaluation script
   ```bash
   python evaluate.py
   ```

#### Current Performance

- **Average Inference Score**: `4.706666666666667` ([detailed results](./evaluation_results.json))
- **Baseline Assessment**: This score represents a **Strong Baseline Performance** for the fine-tuned model
- **Evaluation Dataset**: 150 examples from Evol-Instruct test set


#### ⚠️⚠️⚠️ Important Evaluation Notes ⚠️⚠️⚠️

> **Data Leakage Concern**: The evaluation currently uses `test_set_evol_instruct_150.json` due to the unavailability of the official `evol_instruct_gt.json` ground truth file. **The strong performance may be partially attributed to the model being evaluated on data from the same distribution as the test set**, which could introduce potential overfitting bias since the test data structure was known during development.
> 
> **Recommendation**: For production evaluation, use a completely held-out dataset that was not referenced during model development to obtain more reliable performance metrics.

## Output Files

- `pred.json`: Model predictions on test set
- `training_config.json`: Training hyperparameters used
- `lora_model/`: Saved LoRA adapter weights
- `evaluation_results.json`: Evaluation scores and comparisons

## Tips for Better Performance

1. **LoRA Rank**: Higher ranks (64, 128) allow more expressive power but increase parameters
2. **Learning Rate**: Start with 2e-4 and adjust based on training curves
3. **Data Selection**: Use advanced sorting to prioritize high-quality, optimally-sized conversations
4. **Temperature**: Lower values (0.7-1.0) for more focused outputs, higher (1.5+) for creativity
5. **Curriculum Learning**: Train progressively from simple to complex examples

## License

This project is part of ML Spring 2025 coursework. Please follow academic integrity guidelines.
