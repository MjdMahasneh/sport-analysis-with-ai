"""
config.py — Central configuration for SportVision.

All tuneable constants live here. No magic numbers should exist elsewhere in
the codebase — import from this module instead.
"""

# ── Model weights ─────────────────────────────────────────────────────────────
POSE_MODEL     = "yolo11n-pose.pt"   # YOLOv11 nano-pose; auto-downloaded on first run
TRACKER_CONFIG = "bytetrack.yaml"   # ByteTrack config bundled with Ultralytics

# ── Detection / OCR confidence thresholds ────────────────────────────────────
DEFAULT_CONF = 0.25   # minimum YOLO detection confidence (0–1)
OCR_CONF     = 0.30   # minimum EasyOCR confidence to include a result

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en"]

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.8    # slightly creative; keep grounded in detection data
GROQ_MAX_TOKENS  = 200    # 2–3 sentences is plenty

# ── Trail visuals ─────────────────────────────────────────────────────────────
DEFAULT_TRAIL_LEN     = 40   # past positions drawn per tracked object
TRAJECTORY_N_FRAMES   = 20   # frames ahead to project per player
TRAJECTORY_VEL_WINDOW = 6    # trailing frames used for velocity estimate

# ── Team clustering ───────────────────────────────────────────────────────────
TEAM_COLOR_HISTORY = 15   # rolling observation window for jersey-colour averaging
# On-screen highlight colours for bounding boxes (BGR).
# These are visual affordances and are independent of detected jersey colour.
TEAM_A_BOX_BGR     = ( 40, 180, 255)   # orange
TEAM_B_BOX_BGR     = (255,  80,  40)   # blue
TEAM_NAMES         = ["A", "B"]

# Jersey colour extraction settings
JERSEY_RESIZE       = 32    # resize jersey crop to NxN before K-Means (was 16)
JERSEY_KP_CONF      = 0.4   # min keypoint confidence to use pose-guided torso crop
# HSV skin-tone exclusion ranges (works across ethnicities)
JERSEY_SKIN_H_MAX   = 25    # upper hue bound for skin (0-180 OpenCV scale)
JERSEY_SKIN_H_MIN   = 160   # lower hue bound for wrap-around red skin
JERSEY_SKIN_S_MIN   = 15    # min saturation to be considered skin
JERSEY_SKIN_S_MAX   = 170   # max saturation to be considered skin
JERSEY_SKIN_V_MIN   = 50    # min value (brightness) to be considered skin
JERSEY_DARK_V_MAX   = 30    # pixels darker than this are shadows / background

# ── Streamlit UI ──────────────────────────────────────────────────────────────
TRAIL_LEN_MIN     = 10
TRAIL_LEN_MAX     = 80
TRAIL_LEN_DEFAULT = DEFAULT_TRAIL_LEN
TRAIL_LEN_STEP    = 5

