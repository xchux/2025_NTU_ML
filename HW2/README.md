# ML2025Spring_hw2

## Project Description
This project is a Machine Learning homework assignment (ML2025Spring_hw2_public) based on the AIDE (AI-Driven Exploration in the Space of Code) framework. The goal is to develop a model that predicts the probability of testing positive for COVID-19 on day 3, using survey data from previous days. The project leverages an agent powered by a Large Language Model (LLM) to automate code drafting, improvement, and debugging for ML solutions.

## Directory Structure
```
├── ML2025Spring_hw2_public.ipynb      # Main Jupyter notebook
├── best_solution.py                   # Best solution script (auto-generated)
├── good_solution_*.py                 # Other good solution scripts
├── submission.csv                     # Output predictions for submission
├── ML2025Spring-hw2-public/           # ‼️ Please download from google drive ‼️
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Test data
│   └── sample_submission.csv          # Submission format example
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Python project metadata
├── Dockerfile                         # CUDA-enabled reproducible environment
└── ... (LLM model files, etc.)
```

## Data Overview
- **train.csv**: Contains survey features and COVID-19 test results for training.
- **test.csv**: Contains survey features for which predictions are to be made.
- **sample_submission.csv**: Shows the required format for prediction submissions.

Typical columns include demographic info, symptoms, and previous test results. The target is the probability of testing positive on day 3.

## Setup Instructions

### Using Docker (Recommended)
A CUDA-enabled Ubuntu environment is provided for reproducibility.

```bash
docker build -t ml2025_hw2 .
docker run --gpus all -it -v $(pwd):/workspace ml2025_hw2
```
- This mounts your workspace and enables GPU acceleration if available.

## Running Solution Scripts
You can run the best or good solution scripts directly to generate predictions:

```bash
python best_solution.py
```
- These scripts will read `test.csv` and output predictions to `submission.csv` in the required format.

## Generating Predictions
1. Ensure `test.csv` is present in the `ML2025Spring-hw2-public/` directory.
2. Run the desired solution script (see above).
3. The predictions will be saved to `submission.csv`.

## LLM Model Files
- Large LLM model files (e.g., `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`) are used for agent-driven code generation and improvement.
- When using `deepseek-coder-6.7b-instruct.Q5_0.gguf` had encountered blank `best_solution.py` and complete `submission.csv`
- When using `qwen2.5-coder-7b-instruct-q8_0.gguf` beware of docker resource. In mine experience at least 10 GB RAM and high performance disk.

## Notes
- All dependencies are listed in `requirements.txt` and `pyproject.toml`.
- The Dockerfile ensures a consistent, GPU-enabled environment for reproducibility.
- For best results, use the provided Docker setup or match the Python and package versions exactly.
- ‼️It is still difficult to obtain reproducible and stable results. ‼️

## Best result
Private Score: 0.94748

Public score: 1.01036

## Contact
For questions or issues, please refer to your course instructor or teaching assistant.
