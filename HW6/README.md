# ML2025 Homework 6 - Fine-tuning leads to Forgetting

This project demonstrates the phenomenon of catastrophic forgetting in large language models by fine-tuning a pre-trained Llama-3.2-1B-Instruct model on the GSM8K dataset. The goal is to observe how fine-tuning on a specific task (mathematical reasoning) affects the model's performance on previously learned knowledge, particularly safety-related aspects evaluated via the AILuminate dataset.

## Project Structure

- `ml2025hw6.ipynb`: Main Jupyter notebook containing the code for data preparation, model fine-tuning, inference, and evaluation.
- `sft/`: Directory containing fine-tuned model checkpoints (e.g., checkpoint-500, checkpoint-1000, etc.).
- `pyproject.toml`: Project configuration file.
- `Dockerfile`: Docker configuration for the dev container.
- `devcontainer_template/`: Rename to .devcontainer for the dev container setup.

## Requirements

- Python 3.11+
- CUDA-compatible GPU (for model training and inference)
- Hugging Face account with access to Llama models
- Internet connection for downloading datasets and models

## Setup

1. Hugging Face Login: Log in to Hugging Face to access the Llama model. Replace the token in the notebook with your own:
    ```
    huggingface-cli login --token "your_huggingface_token"
    ```

## Usage

1. **Key Parameters**:
    - TRAIN_N_SHOT: Number of few-shot examples for training (default: 8).
    - TEST_N_SHOT: Number of few-shot examples for testing (default: 8).
    - adapter_path: Path to the checkpoint to evaluate (default: 'sft/checkpoint-1868').
    - Evaluation: The notebook computes accuracy on GSM8K and generates predictions for AILuminate. Results are saved to {STUDENT_ID}.txt.

## Results

‼️Because it is not an official verification tool, the score may be slightly inaccurate.‼️

Current Accuracy = 0.379

Just reached the **medium** baseline.

## References

  - [Reproducing Llama 3's Performance on GSM8K](https://medium.com/@sewoong.lee/how-to-reproduce-llama-3s-performance-on-gsm-8k-e0dce7fe9926)
  - [AILuminate Repository](https://github.com/mlcommons/ailuminate/tree/main)
  - Hugging Face PEFT and Transformers documentation.
