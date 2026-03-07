import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

BASE_DIR = Path("ai-lab/model-comparison")
PROMPTS_PATH = BASE_DIR / "prompts.json"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prompts(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("prompts.json must contain a JSON array of strings.")

    return prompts


def run_model(model: str, prompt: str) -> dict:
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    text = response.output[0].content[0].text
    return {
        "model": model,
        "prompt": prompt,
        "response_text": text,
    }


def main() -> None:
    prompts = load_prompts(PROMPTS_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for i, prompt in enumerate(prompts, start=1):
        print("\n" + "#" * 60)
        print(f"PROMPT {i}: {prompt}")
        print("#" * 60)

        for model in MODELS:
            print("\n" + "=" * 40)
            print(f"MODEL: {model}")
            print("=" * 40)

            result = run_model(model, prompt)
            results.append(result)

            print(result["response_text"])
            print()

    output_path = OUTPUT_DIR / f"comparison_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "models": MODELS,
                "prompts": prompts,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()