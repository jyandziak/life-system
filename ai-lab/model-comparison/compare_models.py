import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = "Explain what an agentic AI workflow is in one paragraph."

MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

OUTPUT_DIR = Path("ai-lab/model-comparison/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for model in MODELS:
        print("\n" + "=" * 40)
        print(f"MODEL: {model}")
        print("=" * 40)

        result = run_model(model, PROMPT)
        results.append(result)

        print(result["response_text"])
        print()

    output_path = OUTPUT_DIR / f"comparison_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()