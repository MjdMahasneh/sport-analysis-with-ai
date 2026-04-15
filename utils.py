"""
utils.py — Shared visual helpers for SportVision.

Provides:
  - Color palette keyed by track ID
  - Trail drawing   : fading motion-history lines per tracked object
  - Trajectory      : linear velocity extrapolation ahead of each player
  - Team clustering : K-means on jersey color → Team A / Team B assignment
"""

from collections import deque
import cv2
import numpy as np

import config

# ── Track-ID color palette ────────────────────────────────────────────────────
TRACK_COLORS: list[tuple[int, int, int]] = [
    (56,  56,  255), (151, 157, 255), (31,  112, 255), (29,  178, 255),
    (49,  210, 207), (10,  249,  72), (23,  204, 146), (134, 219,  61),
    (52,  147,  26), (187, 212,   0), (168, 153,  44), (255, 194,   0),
    (147,  69,  52), (255, 115, 100), (236,  24,   0), (255,  56, 132),
    (133,   0,  82), (255,  56, 203), (200, 149, 255), (199,  55, 255),
]

# Pull team colours and names from central config
TEAM_BOX_COLORS = [config.TEAM_A_BOX_BGR, config.TEAM_B_BOX_BGR]
TEAM_NAMES      = config.TEAM_NAMES


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR color for a given ByteTrack track ID."""
    return TRACK_COLORS[abs(track_id) % len(TRACK_COLORS)]


# ── Trail helpers ─────────────────────────────────────────────────────────────

def update_trail(
    trail_history: dict[int, deque],
    track_id: int,
    cx: int,
    cy: int,
    max_len: int = 40,
) -> None:
    """Push the current bbox centre into the trail deque for this track_id."""
    if track_id not in trail_history:
        trail_history[track_id] = deque(maxlen=max_len)
    trail_history[track_id].append((cx, cy))


def draw_trails(frame: np.ndarray, trail_history: dict[int, deque]) -> np.ndarray:
    """Render fading motion trails — dim+thin (oldest) → bright+thick (newest)."""
    for track_id, trail in trail_history.items():
        pts = list(trail)
        n = len(pts)
        base = get_track_color(track_id)
        for i in range(1, n):
            alpha = i / n
            color = tuple(int(c * alpha) for c in base)
            cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(4 * alpha)))
    return frame


# ── Trajectory projection ─────────────────────────────────────────────────────

def _compute_projected_points(
    trail: deque,
    n_frames: int = config.TRAJECTORY_N_FRAMES,
    velocity_window: int = config.TRAJECTORY_VEL_WINDOW,
) -> list[tuple[int, int]]:
    pts = list(trail)
    if len(pts) < 3:
        return []
    recent = pts[-min(velocity_window, len(pts)):]
    span = max(len(recent) - 1, 1)
    dx = (recent[-1][0] - recent[0][0]) / span
    dy = (recent[-1][1] - recent[0][1]) / span
    if abs(dx) < 0.8 and abs(dy) < 0.8:
        return []
    cx, cy = pts[-1]
    return [(int(cx + dx * i), int(cy + dy * i)) for i in range(1, n_frames + 1)]


def draw_trajectories(
    frame: np.ndarray,
    trail_history: dict[int, deque],
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Draw fading dotted projection lines ahead of each tracked object."""
    for track_id, trail in trail_history.items():
        projected = _compute_projected_points(trail)
        if not projected:
            continue
        base = get_track_color(track_id)
        n = len(projected)
        for i, (px, py) in enumerate(projected):
            if not (0 <= px < frame_w and 0 <= py < frame_h):
                break
            alpha = 1.0 - (i / n)
            color = tuple(int(c * alpha * 0.85) for c in base)
            cv2.circle(frame, (px, py), max(2, int(5 * alpha)), color, -1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  TEAM CLUSTERING  (LAB colour space + skin/shadow masking + keypoint crop)
# ══════════════════════════════════════════════════════════════════════════════

def _bgr_to_lab(bgr_arr: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) float32 BGR array (0-255 range) to uint8 LAB."""
    n = len(bgr_arr)
    img = bgr_arr.clip(0, 255).astype(np.uint8).reshape(n, 1, 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(n, 3).astype(np.float32)


def _lab_to_bgr(lab_arr: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) float32 LAB array back to (N, 3) float32 BGR."""
    n = len(lab_arr)
    img = lab_arr.clip(0, 255).astype(np.uint8).reshape(n, 1, 3)
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR).reshape(n, 3).astype(np.float32)


def get_jersey_color(
    frame_bgr: np.ndarray,
    bbox: tuple,
    keypoints: list[dict] | None = None,
) -> np.ndarray:
    """
    Extract the dominant jersey colour from a player bounding box.

    Improvements over the naive approach
    -------------------------------------
    1. **Keypoint-guided crop** — if shoulder/hip keypoints are available and
       confident (≥ JERSEY_KP_CONF), the crop is bounded exactly by those
       landmarks.  Falls back to a fixed-fraction window otherwise.
    2. **32×32 resize** — 4× more pixels fed into K-Means than the old 16×16.
    3. **Skin-tone masking** — HSV ranges for human skin are excluded so that
       exposed face/neck/arms do not contaminate the jersey colour estimate.
    4. **Shadow / glare masking** — near-black (shadow) pixels are removed.
    5. **LAB-space K-Means** — dominant colour is extracted in the perceptually
       uniform CIE LAB space, then converted back to BGR for display.

    Returns
    -------
    np.ndarray, shape (3,) — dominant jersey colour as BGR float32.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)

    # ── 1. Determine crop region ──────────────────────────────────────────────
    crop_coords: tuple | None = None
    if keypoints:
        try:
            by_name = {kp["name"]: kp for kp in keypoints}
            thr = config.JERSEY_KP_CONF
            sho = [kp for name in ("left shoulder", "right shoulder")
                   if (kp := by_name.get(name)) and kp["confidence"] >= thr]
            hip = [kp for name in ("left hip", "right hip")
                   if (kp := by_name.get(name)) and kp["confidence"] >= thr]
            if sho and hip:
                all_x  = [kp["x"] for kp in sho + hip]
                kp_y1  = max(int(min(kp["y"] for kp in sho)), y1)
                kp_y2  = min(int(max(kp["y"] for kp in hip)), y2)
                kp_x1  = max(int(min(all_x)) - 8, x1)
                kp_x2  = min(int(max(all_x)) + 8, x2)
                if (kp_x2 - kp_x1) > 5 and (kp_y2 - kp_y1) > 5:
                    crop_coords = (kp_x1, kp_y1, kp_x2, kp_y2)
        except Exception:
            pass

    if crop_coords is None:
        # Fallback: fixed window — tighter top margin to skip more of the head
        cy1 = y1 + max(int(h * 0.22), 1)
        cy2 = y1 + max(int(h * 0.65), cy1 + 1)
        cx1 = x1 + max(int(w * 0.18), 1)
        cx2 = x2 - max(int(w * 0.18), 1)
        crop_coords = (cx1, cy1, cx2, cy2)

    cx1, cy1, cx2, cy2 = crop_coords
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    # ── 2. Resize and build pixel masks ──────────────────────────────────────
    sz    = config.JERSEY_RESIZE
    small = cv2.resize(crop, (sz, sz))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Skin: broad hue range (wraps around 0/180) + saturation / value bounds
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    is_skin = (
        ((h_ch <= config.JERSEY_SKIN_H_MAX) | (h_ch >= config.JERSEY_SKIN_H_MIN))
        & (s_ch >= config.JERSEY_SKIN_S_MIN)
        & (s_ch <= config.JERSEY_SKIN_S_MAX)
        & (v_ch >= config.JERSEY_SKIN_V_MIN)
    )
    is_shadow = v_ch < config.JERSEY_DARK_V_MAX   # near-black → shadow/background

    bad = (is_skin | is_shadow).flatten()

    # ── 3. K-Means in LAB space on valid pixels ───────────────────────────────
    small_lab  = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pixels_lab = small_lab.reshape(-1, 3).astype(np.float32)
    pixels_bgr = small.reshape(-1, 3).astype(np.float32)

    valid = ~bad
    lab_in = pixels_lab[valid] if valid.sum() >= 10 else pixels_lab
    bgr_in = pixels_bgr[valid] if valid.sum() >= 10 else pixels_bgr

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, center_lab = cv2.kmeans(
        lab_in, 1, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    # Convert LAB centroid back to BGR for display compatibility
    center_bgr = cv2.cvtColor(
        center_lab.clip(0, 255).astype(np.uint8).reshape(1, 1, 3),
        cv2.COLOR_LAB2BGR,
    ).reshape(3).astype(np.float32)
    return center_bgr


def cluster_teams(
    jersey_colors: list[np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """
    Split players into two teams using K-Means (k=2) in **CIE LAB** space.

    Clustering in LAB is far more reliable than BGR because equal Euclidean
    distances in LAB correspond to equal perceived colour differences — so
    a red jersey and a blue jersey are always well-separated regardless of
    their shared brightness.

    Labels are normalised so that Team 0 always corresponds to the cluster
    with the higher LAB L* (i.e. the lighter jersey), giving a consistent
    team identity across frames even if K-Means labels flip.

    Returns
    -------
    team_labels : list[int]        — 0 or 1 per player
    centers_bgr : np.ndarray (2,3) — cluster centres in BGR (for display)
    """
    n = len(jersey_colors)
    fallback_centers = np.array([[200, 180, 255], [255, 80, 40]], dtype=np.float32)

    if n == 0:
        return [], fallback_centers
    if n == 1:
        return [0], fallback_centers

    colors_bgr = np.array(jersey_colors, dtype=np.float32)
    colors_lab = _bgr_to_lab(colors_bgr)   # perceptually uniform feature space

    if n == 2:
        # Normalise: higher L* (lighter jersey) → Team 0
        if colors_lab[0, 0] < colors_lab[1, 0]:
            return [1, 0], colors_bgr[[1, 0]]
        return [0, 1], colors_bgr

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_lab = cv2.kmeans(
        colors_lab, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten().tolist()

    # Convert centers back to BGR for the swatch display
    centers_bgr = _lab_to_bgr(centers_lab)

    # Normalise: Team 0 = lighter cluster (stable across frame re-runs)
    if centers_lab[0, 0] < centers_lab[1, 0]:
        labels = [1 - lbl for lbl in labels]
        centers_bgr = centers_bgr[[1, 0]]

    return labels, centers_bgr


def draw_team_overlay(
    frame: np.ndarray,
    detections: list[dict],
    team_labels: list[int],
    centers: np.ndarray | None = None,
) -> np.ndarray:
    """
    Draw team-coloured bounding boxes + badge labels on `frame` (BGR, in-place).

    Each badge shows "Team A" / "Team B" in a highlight colour plus a small
    swatch of the detected jersey colour so viewers can see the raw colour
    that drove the clustering decision.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det, team in zip(detections, team_labels):
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        team_idx = team % 2
        box_color = TEAM_BOX_COLORS[team_idx]
        label_text = f"Team {TEAM_NAMES[team_idx]}"

        # Thick team-coloured bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        # Badge dimensions
        scale, thick = 0.55, 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thick)
        bx1, by1 = x1, max(y1 - th - baseline - 8, 0)
        bx2, by2 = x1 + tw + 26, y1   # extra room for colour swatch

        # Filled badge background
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, -1)

        # Badge text
        cv2.putText(
            frame, label_text,
            (bx1 + 4, by2 - baseline - 2),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )

        # Jersey colour swatch (rightmost 16 px of badge)
        if centers is not None and len(centers) > team_idx:
            swatch_color = tuple(int(c) for c in centers[team_idx])
            cv2.rectangle(frame, (bx2 - 16, by1 + 2), (bx2 - 2, by2 - 2), swatch_color, -1)
            cv2.rectangle(frame, (bx2 - 16, by1 + 2), (bx2 - 2, by2 - 2), (255, 255, 255), 1)

    return frame


class TeamTracker:
    """
    Maintains stable team assignments across video frames.

    Algorithm
    ---------
    1. Accumulate a short rolling window of jersey-colour observations
       per ByteTrack track_id.
    2. Every frame, compute per-track average colour and re-run K-means(k=2).
    3. Newly-seen tracks with insufficient history are assigned by nearest
       cluster centre rather than re-clustering, preventing label instability.
    """

    def __init__(self, color_history_len: int = config.TEAM_COLOR_HISTORY):
        self.color_history:   dict[int, deque]    = {}   # track_id → recent colours
        self.team_assignments: dict[int, int]     = {}   # track_id → team 0/1
        self.cluster_centers: np.ndarray | None   = None
        self.history_len = color_history_len

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict],
        track_ids: list[int],
        keypoints_list: list[list[dict] | None] | None = None,
    ) -> dict[int, int]:
        """
        Update colour history, re-cluster, and return track_id→team mapping.

        Parameters
        ----------
        keypoints_list : optional list (same length as detections) of per-player
                         COCO keypoint dicts.  Forwarded to get_jersey_color so
                         shoulder/hip landmarks can guide the torso crop.
        """
        # 1. Record jersey colour for each visible track
        for i, (det, track_id) in enumerate(zip(detections, track_ids)):
            kps = keypoints_list[i] if keypoints_list and i < len(keypoints_list) else None
            color = get_jersey_color(
                frame_bgr, (det["x1"], det["y1"], det["x2"], det["y2"]), kps
            )
            if track_id not in self.color_history:
                self.color_history[track_id] = deque(maxlen=self.history_len)
            self.color_history[track_id].append(color)

        # 2. Only cluster tracks with at least 3 colour observations
        stable = {
            tid: np.mean(list(cols), axis=0)
            for tid, cols in self.color_history.items()
            if len(cols) >= 3
        }

        if len(stable) >= 2:
            tids   = list(stable.keys())
            colors = [stable[t] for t in tids]
            labels, centers = cluster_teams(colors)
            self.cluster_centers = centers
            for tid, label in zip(tids, labels):
                self.team_assignments[tid] = label

        # 3. Assign new / unstable tracks to nearest existing cluster centre
        if self.cluster_centers is not None:
            for track_id in track_ids:
                if track_id not in self.team_assignments:
                    if self.color_history.get(track_id):
                        avg = np.mean(list(self.color_history[track_id]), axis=0)
                        dists = [np.linalg.norm(avg - c) for c in self.cluster_centers]
                        self.team_assignments[track_id] = int(np.argmin(dists))

        return self.team_assignments

