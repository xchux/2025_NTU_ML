
#!/usr/bin/env python3
"""Score pred.json for ML2025 HW9 model merging.

Features:
1. Extract final MCQA option (A/B/C/D) from free-form responses using regex patterns.
2. Load ground truth from local files if `correct_option` exists.
3. Fallback to Hugging Face dataset for ground truth labels when local files have no labels.
4. Report ARC/GSM8K/overall accuracy and baseline pass status.
5. Print qualitative samples for manual inspection.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASELINES = {
    "Simple": {"ARC": 0.49, "GSM8K": 0.38},
    "Medium": {"ARC": 0.53, "GSM8K": 0.42},
    "Strong": {"ARC": 0.56, "GSM8K": 0.48},
}


@dataclass
class Example:
    example_id: str
    task_name: str
    correct_option: Optional[str]
    question: str


def normalize_option(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    # Accept formats like "A", "(A)", "option A", "Answer: C", "A.".
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return None


def extract_option(response: str) -> Optional[str]:
    if not response:
        return None

    patterns = [
        r"the\s+answer\s+is\s*(?:option\s*)?[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?",
        r"the\s+correct\s+answer\s+is\s*(?:option\s*)?[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?",
        r"therefore,?\s*(?:the\s*)?(?:option\s*)?[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?",
        r"answer\s*[:\-]\s*(?:option\s*)?[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?",
        r"option\s*([ABCD])\s+is\s+correct",
    ]

    lowered = response.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: prefer the last standalone option marker in text.
    fallback_matches = re.findall(r"\b([ABCD])\b", response.upper())
    if fallback_matches:
        return fallback_matches[-1]
    return None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pred(path: Path) -> Dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object mapping id -> response")
    return {str(k): str(v) for k, v in data.items()}


def load_examples(path: Path, default_task_name: str) -> List[Example]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")

    examples = []
    for row in data:
        if not isinstance(row, dict):
            continue
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            continue
        task_name = str(row.get("task_name") or default_task_name)
        question = str(row.get("question") or "")
        examples.append(
            Example(
                example_id=example_id,
                task_name=task_name,
                correct_option=normalize_option(row.get("correct_option")),
                question=question,
            )
        )
    return examples


def inject_ground_truth_from_hf(
    examples: List[Example], dataset_id: str, split_name: str
) -> int:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Could not import datasets. Install it with: pip install datasets"
        ) from exc

    hf_data = load_dataset(dataset_id, split=split_name)
    by_id: Dict[str, Optional[str]] = {}
    for row in hf_data:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue
        option = normalize_option(row.get("correct_option"))
        if option is None:
            for k in ("answer", "label", "gold", "target"):
                option = normalize_option(row.get(k))
                if option is not None:
                    break
        by_id[rid] = option

    filled = 0
    for ex in examples:
        if ex.correct_option is None:
            candidate = by_id.get(ex.example_id)
            if candidate is not None:
                ex.correct_option = candidate
                filled += 1
    return filled


def build_task_map(examples: Iterable[Example]) -> Dict[str, List[Example]]:
    out: Dict[str, List[Example]] = {}
    for ex in examples:
        out.setdefault(ex.task_name, []).append(ex)
    return out


def calc_stats(
    task_examples: List[Example], pred_map: Dict[str, str]
) -> Tuple[int, int, int, int, List[Tuple[Example, Optional[str], str]]]:
    total = len(task_examples)
    has_gt = 0
    extracted = 0
    correct = 0
    mismatches: List[Tuple[Example, Optional[str], str]] = []

    for ex in task_examples:
        gt = ex.correct_option
        if gt is None:
            continue
        has_gt += 1
        response = pred_map.get(ex.example_id, "")
        pred_opt = extract_option(response)
        if pred_opt is not None:
            extracted += 1
        if pred_opt == gt:
            correct += 1
        else:
            mismatches.append((ex, pred_opt, response))

    return total, has_gt, extracted, correct, mismatches


def summarize_repetition(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return "empty"
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.35:
        return "high repetition"
    if unique_ratio < 0.55:
        return "moderate repetition"
    return "low repetition"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pred.json for ARC + GSM8K")
    parser.add_argument("--pred", default="pred.json", help="Path to pred.json")
    parser.add_argument("--arc", default="ARC.json", help="Path to ARC questions JSON")
    parser.add_argument("--gsm8k", default="GSM8K.json", help="Path to GSM8K questions JSON")
    parser.add_argument(
        "--dataset-id",
        default="MonicaHuang/ML2025_HW9",
        help="Hugging Face dataset id used to fetch correct_option if missing locally",
    )
    parser.add_argument(
        "--disable-hf",
        action="store_true",
        help="Disable Hugging Face fallback loading",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="How many mismatch samples to show per task",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    arc_path = Path(args.arc)
    gsm8k_path = Path(args.gsm8k)

    pred_map = load_pred(pred_path)
    arc_examples = load_examples(arc_path, default_task_name="ARC")
    gsm8k_examples = load_examples(gsm8k_path, default_task_name="GSM8K")

    # Fill missing labels from HF if local JSON does not contain correct_option.
    missing_arc = sum(1 for x in arc_examples if x.correct_option is None)
    missing_gsm = sum(1 for x in gsm8k_examples if x.correct_option is None)
    if not args.disable_hf and (missing_arc > 0 or missing_gsm > 0):
        print("[Info] Missing local labels detected. Trying Hugging Face fallback...")
        filled_arc = inject_ground_truth_from_hf(arc_examples, args.dataset_id, "ARC")
        filled_gsm = inject_ground_truth_from_hf(gsm8k_examples, args.dataset_id, "GSM8K")
        print(
            f"[Info] Filled labels from HF: ARC={filled_arc}, GSM8K={filled_gsm}"
        )

    task_map = build_task_map(arc_examples + gsm8k_examples)
    task_order = ["ARC", "GSM8K"]

    print("\n========== Scoring Report ==========")
    overall_correct = 0
    overall_with_gt = 0

    for task in task_order:
        examples = task_map.get(task, [])
        total, has_gt, extracted, correct, mismatches = calc_stats(examples, pred_map)
        acc = (correct / has_gt) if has_gt else 0.0

        overall_correct += correct
        overall_with_gt += has_gt

        print(f"\n[{task}]")
        print(f"- total questions: {total}")
        print(f"- with ground truth: {has_gt}")
        print(f"- extracted options: {extracted}")
        print(f"- correct: {correct}")
        print(f"- accuracy: {acc:.4f} ({acc*100:.2f}%)")

        for level, thresholds in BASELINES.items():
            target = thresholds[task]
            passed = acc >= target
            verdict = "PASS" if passed else "FAIL"
            print(f"- {level} baseline ({target*100:.0f}%): {verdict}")

        if args.show_samples > 0 and mismatches:
            print("- qualitative mismatch samples:")
            for ex, pred_opt, response in mismatches[: args.show_samples]:
                snippet = response.replace("\n", " ").strip()
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
                rep = summarize_repetition(response)
                print(
                    "  * id={}; gt={}; pred={}; repetition={}; response_snippet={}".format(
                        ex.example_id,
                        ex.correct_option,
                        pred_opt,
                        rep,
                        snippet,
                    )
                )

    overall_acc = (overall_correct / overall_with_gt) if overall_with_gt else 0.0
    print("\n[Overall]")
    print(f"- with ground truth: {overall_with_gt}")
    print(f"- correct: {overall_correct}")
    print(f"- accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    if overall_with_gt != 400:
        print(
            "[Warning] Ground-truth size is not 400. "
            f"Current scored denominator is {overall_with_gt}."
        )


if __name__ == "__main__":
    main()