# HW8 Model Editing

## Environment
This code works on Google Colab or similar Jupyter environment with GPU support (T4 or better).
It is also compatible with the provided Dev Container environment.

## Dependencies
The notebook installs necessary dependencies. If running locally, ensure you have:
- `torch`
- `transformers`
- `datasets`
- `matplotlib`
- `scipy`
- `numpy`

## Files
- `Homework_8_Model_Editing.ipynb`: The main notebook containing the implementation of ROME and MEMIT model editing.
- `modify_notebook.py`: Helper script used to apply modifications (optional).
- `data/HW8_data.json`: Dataset for model editing.

## Usage
1.  Open `Homework_8_Model_Editing.ipynb`.
2.  Run the cells in order.
3.  The notebook is pre-configured to:
    -   Use `gpt2-xl`.
    -   Perform Single Editing using ROME method with a custom request ("The Eiffel Tower is located in" -> "Rome").
    -   Perform Multiple Editing using ROME and MEMIT on the full dataset (80 examples).

## Modifications
-   Implemented the missing logic in `apply_rome_to_model` to calculate the update matrix `W' = W + v k^T`.
-   Updated the Single Editing Request to use a custom prompt as required by the regulations.
-   Configured the notebook to use ROME instead of FT for Single Editing.
-   Enabled the full dataset (80 samples) for Multiple Editing.

## References
-   [ROME Paper](https://arxiv.org/pdf/2202.05262)
-   [MEMIT Paper](https://arxiv.org/pdf/2210.07229)
-   [ROME GitHub](https://github.com/kmeng01/rome)
-   [MEMIT GitHub](https://github.com/kmeng01/memit)
