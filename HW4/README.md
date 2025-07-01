# ML2025 Spring HW4: Training Transformer on Pokémon Images

## Description
This assignment focuses on pretraining a transformer decoder-only architecture for next-token prediction, specifically applied to Pokémon images. The goal is to reconstruct images by predicting pixel color sequences using models such as GPT-2 or Qwen3.

- [TA's Slide](https://docs.google.com/presentation/d/1ga0d43mWyrfHjdkp7FG3iWcEKTGr8CdkifYhhi4LBY8/edit?usp=sharing)
- Contact: ntu-ml-2025-spring-ta@googlegroups.com

## Dataset
- Pokémon dataset: [lca0503/ml2025-hw4-pokemon](https://huggingface.co/datasets/lca0503/ml2025-hw4-pokemon)
- Colormap: [lca0503/ml2025-hw4-colormap](https://huggingface.co/datasets/lca0503/ml2025-hw4-colormap)


## Data Preparation
- The dataset consists of pixel color sequences for Pokémon images.
- The `PixelSequenceDataset` class handles train/dev/test splits and prepares data for the model.


## Evaluation
1. **Create Valid Img**: 
    -  Execute `python3 preprocess.py gen-valid-img` to save dev img to ground_truth.txt.
    -  Execute `python preprocess.py create-img -i ground_truth.txt -o validation` for conver ground_truth.txt to img which will use for `pytorch-fid`.
2. **Create Classifier**: Execute `python3 preprocess.py gen-classifier`
3. **Run the notebook** (`ML2025_Spring_HW4.ipynb`) and get reconstructed_results.txt
2. **Get valid data - FID**
    - `pytorch-fid reconstructed/ validation/`
3. **Get valid data - PDR**
    - Run `python3 pdr_score.py` to get PDR.


## Result
‼️Because it is not an official verification tool, the score may be slightly inaccurate.‼️

FID:  76.22972546960304

PDR (Accuracy) score: 0.9625

Clsoe to Strong baseline❗
