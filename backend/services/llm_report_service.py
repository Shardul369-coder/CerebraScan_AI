import json
import os
import requests
from pathlib import Path

RESULTS_DIR = Path("backend/storage/results")

# Ollama runs locally on port 11434 by default.
# Override with: export OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")  # change to llama3, mistral, etc.


def generate_clinical_report(case_id: str) -> str:

    json_path = RESULTS_DIR / f"{case_id}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Segmentation result not found for case '{case_id}'. "
            "Run the analysis pipeline first."
        )

    with open(json_path, "r") as f:
        features = json.load(f)

    prompt = f"""You are an expert neuroradiologist.

Generate a professional clinical radiology report based on the brain MRI tumor segmentation data below.

SEGMENTATION DATA:
{json.dumps(features, indent=2)}

Write the report using the following format:

Findings:
Impression:
Recommendation:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert neuroradiologist. Write concise, professional clinical reports.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running: 'ollama serve'"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 seconds.")

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {response.text}"
        )

    data = response.json()

    # /api/chat response format
    try:
        return data["message"]["content"]
    except KeyError:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")