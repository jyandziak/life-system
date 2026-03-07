import json
from pathlib import Path

RESULTS_DIR = Path("ai-lab/model-comparison/results")

CRITERIA = [
    "clarity",
    "specificity",
    "practicality",
    "conciseness",
]


def score_response(text: str) -> dict:
    text_lower = text.lower()

    scores = {
        "clarity": 3,
        "specificity": 3,
        "practicality": 3,
        "conciseness": 3,
    }

    if len(text.split()) < 120:
        scores["conciseness"] = 4
    if len(text.split()) > 220:
        scores["conciseness"] = 2

    if any(word in text_lower for word in ["for example", "e.g.", "such as"]):
        scores["specificity"] = 4

    if any(word in text_lower for word in ["use case", "workflow", "step", "process"]):
        scores["practicality"] = 4

    if len(text.split(".")) > 3:
        scores["clarity"] = 4

    return scores


def main() -> None:
    latest_file = sorted(RESULTS_DIR.glob("comparison_*.json"))[-1]

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Evaluating: {latest_file}\n")

    for result in data["results"]:
        scores = score_response(result["response_text"])
        total = sum(scores.values())

        print("=" * 60)
        print(f"MODEL:  {result['model']}")
        print(f"PROMPT: {result['prompt']}")
        print(f"SCORES: {scores} | TOTAL: {total}")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()