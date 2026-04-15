"""
commentary.py — Groq LLM commentary generator for SportVision.

Public API
----------
generate_commentary(detections, ocr_results) -> str

Uses Groq's llama-3.3-70b-versatile model exclusively.
Reads GROQ_API_KEY from the environment — set it before running the app:
    Windows:  $env:GROQ_API_KEY = "your_key_here"
    macOS/Linux: export GROQ_API_KEY="your_key_here"
"""

import os
from groq import Groq

import config

# ── Client singleton ──────────────────────────────────────────────────────────
_client: Groq | None = None


def _get_client() -> Groq:
    """
    Create the Groq client on first call; reuse it afterwards.
    Raises a clear ValueError if GROQ_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set.\n"
                "Get a free key at https://console.groq.com and run:\n"
                "  Windows PowerShell: $env:GROQ_API_KEY = 'your_key'\n"
                "  macOS / Linux:      export GROQ_API_KEY='your_key'"
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(detections: list[dict], ocr_results: list[dict]) -> str:
    """
    Construct a structured prompt from YOLO detections and EasyOCR results.

    Example output:
        DETECTION RESULTS:
        - 5 persons detected
        - 1 sports ball detected

        TEXT IN IMAGE: "7", "23", "BULLS"
    """
    # Count detections per class
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    detection_lines = "\n".join(
        f"  - {cnt} {cls}{'s' if cnt > 1 else ''}"
        for cls, cnt in counts.items()
    )

    # Summarise OCR findings
    if ocr_results:
        quoted = ", ".join(f'"{r["text"]}"' for r in ocr_results)
        ocr_line = f"Text visible in image: {quoted}"
    else:
        ocr_line = "No readable text detected in the image."

    prompt = (
        "Based on the following computer vision analysis of a sports photograph, "
        "write 2–3 sentences of vivid, exciting live sports commentary. "
        "Be specific and energetic. Only reference what the data supports — "
        "do not invent players, scores, or events not implied by the data.\n\n"
        f"DETECTION RESULTS:\n{detection_lines}\n\n"
        f"OCR / TEXT IN IMAGE:\n  {ocr_line}\n\n"
        "Write the commentary now (no intro, no labels — just the commentary):"
    )
    return prompt


# ── Main commentary function ──────────────────────────────────────────────────

def generate_commentary(
    detections: list[dict],
    ocr_results: list[dict],
    model: str = config.GROQ_MODEL,
) -> str:
    """
    Generate AI sports commentary using the Groq API.

    Parameters
    ----------
    detections  : Detection dicts from detector.detect()
                  (each has 'class', 'confidence', 'x1', 'y1', 'x2', 'y2').
    ocr_results : OCR dicts from ocr_reader.read_text()
                  (each has 'text', 'confidence').
    model       : Groq model ID. Default: llama-3.3-70b-versatile.

    Returns
    -------
    Commentary string (2–3 sentences of sports commentary).
    """
    if not detections:
        return (
            "⚠️ No objects were detected — unable to generate meaningful commentary. "
            "Try uploading a clearer sports image."
        )

    client = _get_client()
    prompt = _build_prompt(detections, ocr_results)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class sports commentator known for vivid, "
                    "accurate, and exciting descriptions. You generate commentary "
                    "strictly based on computer vision data — no fabrication. "
                    "Keep responses to 2–3 sentences maximum."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()
