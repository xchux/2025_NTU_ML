# HW9 Model Merging

This README is organized to match the assignment requirements and make it easy for TAs to verify code changes and execution steps.

## 1) PEFT Package Modification Mapping

### 1.1 `replace` Block (Required Format)

If you modified any .py files in the peft package, list them below as "filename: absolute path".

```replace
TODO_1.py: /workspace/peft-ml2025-hw9/src/peft/TODO_1.py
TODO_2.py: /workspace/peft-ml2025-hw9/src/peft/TODO_2.py
```

If you did not modify any .py file in peft for this homework, use:

```replace
NONE.py: /workspace/peft-ml2025-hw9/src/peft/(no_python_file_modified)
```

### 1.2 `main` Block (Inference Entry Point)

```main
/workspace/ML2025_HW9_Model_Merging.ipynb
```

If you eventually run a .py script instead of a notebook, replace this with the absolute path of that main script.

---

## 2) Runtime Environment and Hardware

- Platform: VS Code Dev Container (local Linux container)
- GPU model: TODO (for example T4 / P100 / L4; use CPU if no GPU)
- Python version: 3.11 (provided in pyproject.toml)
- Dependency file: /workspace/pyproject.toml

If you run on Colab or Kaggle, also provide:
- Platform link: TODO
- Runtime setting (GPU/VRAM): TODO

---

## 3) References and Collaboration Record

### 3.1 Reference Sources

- PEFT repository: https://github.com/huggingface/peft
- Course assignment page and slides: TODO
- NTU COOL discussion threads (if any): TODO
- Other references (papers/articles/discussions): TODO

### 3.2 Generative AI Usage Statement

- LLM-assisted work used: Yes
- Tool: GitHub Copilot (GPT-5.3-Codex)
- Assistance scope:
	- README formatting and structure
	- Filling correct_option fields for ARC.json and GSM8K.json
	- Command and process documentation drafting
- Human verification:
	- Final answers and workflow were manually reviewed and confirmed by me
- Conversation record (optional): TODO (add share link here if needed)

### 3.3 Peer Discussion

- None (if any, list student IDs): TODO

---

## 4) Program Execution Guide (Step-by-Step)

This homework currently includes the following main files:

- `ML2025_HW9_Model_Merging.ipynb`
	- Main training and inference workflow.
- `score.py`
	- Scores ARC and GSM8K results from pred.json.
- `ARC.json`, `GSM8K.json`
	- Question datasets (correct_option fields are included).
- `pred.json`
	- Model prediction outputs.

### Execution Order

1. Prepare environment

```bash
cd /workspace
python3 -m pip install -e .
```

2. Run notebook (to generate pred.json)

```bash
# Run in Jupyter or VS Code
/workspace/ML2025_HW9_Model_Merging.ipynb
```

3. Run scoring

```bash
cd /workspace
python3 score.py --pred pred.json --arc ARC.json --gsm8k GSM8K.json
```

4. Validate label completeness (optional)

```bash
cd /workspace
python3 - <<'PY'
import json
for fn in ["ARC.json", "GSM8K.json"]:
		x = json.load(open(fn, "r", encoding="utf-8"))
		ok = sum(1 for i in x if i.get("correct_option") in {"A", "B", "C", "D"})
		print(fn, ok, "/", len(x))
PY
```

---

## 5) Additional Notes

- If TAs require validation via submodule, keep both .gitmodules and the submodule commit pointer in the parent repository.
- Please replace all TODO entries in this README with your actual submission details before final hand-in.
