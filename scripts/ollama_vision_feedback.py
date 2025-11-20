import base64
import json
import os
import argparse
from typing import List, Optional, Dict, Any

import requests


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Try Gemma 3 first; if it fails for vision, fallback to Llama 3.2 Vision
PREFERRED_MODELS: List[str] = [
    "gemma3:27b",
    "llama3.2-vision",
]


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_with_model(model: str, prompt: str, image_paths: List[str]) -> Optional[str]:
    url = f"{OLLAMA_HOST}/api/generate"
    images_b64 = [encode_image_to_base64(p) for p in image_paths]
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        # keep it simple
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, data=json.dumps(payload), headers=headers, timeout=600)
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    # Response may contain 'response' field
    return data.get("response") or data.get("message")


def get_feedback_for_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for img in image_paths:
        prompt = (
            "Analyze this property condition photo and return ONLY a compact JSON object with fields: "
            "file, one_sentence_description, issues (array), score_1_to_5 (integer 1-5), reason."
        )
        response_text: Optional[str] = None
        for model in PREFERRED_MODELS:
            try:
                response_text = generate_with_model(model, prompt, [img])
                if response_text:
                    break
            except Exception:
                continue
        if not response_text:
            results.append({
                "file": os.path.basename(img),
                "error": "No response from available models"
            })
            continue

        parsed: Optional[Any] = None
        try:
            parsed = json.loads(response_text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            # Ensure file name is present
            parsed.setdefault("file", os.path.basename(img))
            results.append(parsed)
        else:
            results.append({
                "file": os.path.basename(img),
                "raw": response_text,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Send property images to Ollama vision model and summarize feedback")
    parser.add_argument("--dir", default="/home/exouser/DSM-property-condition-assessment/Data/randomly-image-test/NHTyp2_selected", help="Directory containing images")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 means all)")
    parser.add_argument("files", nargs="*", help="Explicit image file paths to process (overrides --dir if provided)")
    args = parser.parse_args()

    provided_files = [os.path.abspath(p) for p in args.files if os.path.isfile(p)]
    if provided_files:
        source_files = provided_files
    else:
        base_dir = args.dir
        source_files = sorted(
            [
                os.path.join(base_dir, f)
                for f in os.listdir(base_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    images = source_files if args.limit == 0 else source_files[: max(args.limit, 0)]
    if not images:
        print("No images found.")
        return
    results = get_feedback_for_images(images)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


