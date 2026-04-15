"""
ocr_reader.py — EasyOCR wrapper for SportVision.

Public API
----------
read_text(image, conf_threshold) -> list[str]

The EasyOCR Reader is loaded once (module-level singleton) because
instantiation downloads ~100 MB of model weights — reusing it across
calls is essential for a snappy Streamlit experience.
"""

import easyocr
import numpy as np

import config

# ── Reader singleton ──────────────────────────────────────────────────────────
_reader: easyocr.Reader | None = None


def _get_reader(languages: list[str] | None = None) -> easyocr.Reader:
    """
    Initialise the EasyOCR Reader on first call; return cached instance after.
    Model weights (~100 MB) are downloaded automatically on first run.
    gpu=False keeps the app portable on machines without CUDA.
    """
    global _reader
    if languages is None:
        languages = config.OCR_LANGUAGES
    if _reader is None:
        _reader = easyocr.Reader(languages, gpu=False)
    return _reader


# ── Main OCR function ─────────────────────────────────────────────────────────

def read_text(
    image: np.ndarray,
    conf_threshold: float = config.OCR_CONF,
) -> list[dict]:
    """
    Run EasyOCR on a BGR numpy array (e.g. from cv2.imread).

    Parameters
    ----------
    image           : BGR numpy array (H x W x 3).
    conf_threshold  : Minimum OCR confidence to keep a result (0–1).

    Returns
    -------
    List of dicts, each with:
        - text        (str)   the recognised string
        - confidence  (float) OCR confidence score
    Sorted by confidence descending, de-duplicated.
    """
    reader = _get_reader()

    # EasyOCR expects RGB — flip channels from OpenCV's BGR default
    rgb_image = image[:, :, ::-1]

    # readtext returns [(bounding_box, text, confidence), ...]
    raw_results = reader.readtext(rgb_image)

    results: list[dict] = []
    seen: set[str] = set()

    for (_, text, conf) in raw_results:
        clean = text.strip()
        if conf >= conf_threshold and clean and clean not in seen:
            results.append({"text": clean, "confidence": round(float(conf), 3)})
            seen.add(clean)

    # Sort by confidence so the most certain reads appear first
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results
