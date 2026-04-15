"""
detector.py — YOLOv11-Pose detection + ByteTrack tracking for SportVision.

The pose model (yolo11n-pose.pt) is the ONLY detection model used.
It detects persons and outputs 17 COCO keypoints per person simultaneously,
so every annotated image/frame already shows the skeleton overlay.

Public API
----------
detect(image_path, …)         -> (annotated_bgr, detections, poses)
detect_video(video_path, …)   -> (output_path, all_detections)
summarise_detections(…)       -> {class: count}
summarise_video_detections(…) -> stats dict
"""

from collections import deque

import cv2
import imageio
import numpy as np
from ultralytics import YOLO

import config
from utils import update_trail, draw_trails, draw_trajectories, \
    get_jersey_color, cluster_teams, draw_team_overlay, TeamTracker

# ── COCO 17-keypoint names ────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
]

# ── Model singleton ───────────────────────────────────────────────────────────
# yolo11n-pose.pt is the sole model — it detects persons AND their keypoints.
_model: YOLO | None = None


def _get_model(model_name: str = config.POSE_MODEL) -> YOLO:
    """
    Load yolo11n-pose.pt on first call; return the cached instance thereafter.
    The weights are downloaded automatically by Ultralytics (~6 MB).
    """
    global _model
    if _model is None:
        _model = YOLO(model_name)
    return _model


# ── Shared keypoint extractor ─────────────────────────────────────────────────

def _extract_poses(results) -> list[dict]:
    """Parse Ultralytics keypoint results into a clean list of pose dicts."""
    poses: list[dict] = []
    if results.keypoints is None:
        return poses

    for i, (box, kpts) in enumerate(zip(results.boxes, results.keypoints)):
        kp_xy   = kpts.xy[0].tolist()
        kp_conf = (
            kpts.conf[0].tolist() if kpts.conf is not None
            else [1.0] * len(kp_xy)
        )
        keypoints = [
            {
                "name":       KEYPOINT_NAMES[j] if j < len(KEYPOINT_NAMES) else f"kp{j}",
                "x":          round(float(kp_xy[j][0]), 1),
                "y":          round(float(kp_xy[j][1]), 1),
                "confidence": round(float(kp_conf[j]), 3),
            }
            for j in range(len(kp_xy))
        ]
        poses.append(
            {
                "person":            i + 1,
                "confidence":        round(float(box.conf), 3),
                "keypoints_visible": sum(1 for kp in keypoints if kp["confidence"] > 0.5),
                "keypoints":         keypoints,
            }
        )
    return poses


# ── Image detection + pose ────────────────────────────────────────────────────

def detect(
    image_path: str,
    conf_threshold: float = config.DEFAULT_CONF,
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    """
    Run yolo11n-pose.pt on a single image.

    Returns
    -------
    annotated_bgr : BGR array — skeleton + team-coloured bounding boxes.
    detections    : [{class, confidence, x1, y1, x2, y2}, ...]
    poses         : [{person, confidence, keypoints_visible, keypoints}, ...]
    team_info     : {
                      labels  : list[int]       — 0/1 per detection,
                      centers : np.ndarray(2,3) — jersey BGR cluster centres,
                      counts  : [int, int]      — players per team,
                    }
                    Empty dict if fewer than 2 persons detected.
    """
    model        = _get_model()
    original_bgr = cv2.imread(image_path)          # needed for jersey colour extraction
    results      = model(image_path, conf=conf_threshold, verbose=False)[0]

    # results.plot() draws skeleton + bounding boxes
    annotated_bgr: np.ndarray = results.plot()

    # Build (detection, keypoints) pairs in original YOLO box order so that
    # keypoints stay aligned after the confidence sort below.
    det_kp_pairs: list[tuple[dict, list | None]] = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        det = {
            "class":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }
        kps: list | None = None
        if results.keypoints is not None and i < len(results.keypoints):
            kpt     = results.keypoints[i]
            kp_xy   = kpt.xy[0].tolist()
            kp_conf = kpt.conf[0].tolist() if kpt.conf is not None else [1.0] * len(kp_xy)
            kps = [
                {
                    "name":       KEYPOINT_NAMES[j] if j < len(KEYPOINT_NAMES) else f"kp{j}",
                    "x":          float(kp_xy[j][0]),
                    "y":          float(kp_xy[j][1]),
                    "confidence": float(kp_conf[j]),
                }
                for j in range(len(kp_xy))
            ]
        det_kp_pairs.append((det, kps))

    det_kp_pairs.sort(key=lambda x: x[0]["confidence"], reverse=True)
    detections        = [d  for d, _  in det_kp_pairs]
    keypoints_per_det = [kp for _,  kp in det_kp_pairs]

    poses = _extract_poses(results)

    # ── Team clustering ───────────────────────────────────────────────────────
    team_info: dict = {}
    if len(detections) >= 2 and original_bgr is not None:
        colors = [
            get_jersey_color(original_bgr, (d["x1"], d["y1"], d["x2"], d["y2"]), kps)
            for d, kps in zip(detections, keypoints_per_det)
        ]
        labels, centers = cluster_teams(colors)
        team_info = {
            "labels":  labels,
            "centers": centers,
            "counts":  [labels.count(0), labels.count(1)],
        }
        # Draw team overlay on top of the skeleton annotation
        draw_team_overlay(annotated_bgr, detections, labels, centers)

    return annotated_bgr, detections, poses, team_info


def summarise_detections(detections: list[dict]) -> dict[str, int]:
    """Count detections per class → {'person': 5}."""
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return counts


# ── Video detection + pose + ByteTrack + trails + trajectory ──────────────────

def detect_video(
    video_path: str,
    output_path: str,
    conf_threshold: float = config.DEFAULT_CONF,
    show_trails: bool = True,
    show_trajectory: bool = True,
    trail_length: int = config.DEFAULT_TRAIL_LEN,
    progress_callback=None,
) -> tuple[str, list[dict]]:
    """
    Run yolo11n-pose.pt + ByteTrack on a video, frame by frame.

    Each output frame shows:
      • Bounding box + track ID (from ByteTrack)
      • Full 17-keypoint skeleton overlay (from yolo11n-pose.pt)
      • Fading motion trail per player  (if show_trails=True)
      • Linear trajectory projection    (if show_trajectory=True)

    Output is encoded as H.264 via imageio-ffmpeg for browser compatibility.

    Returns
    -------
    output_path    : Path to the annotated H.264 MP4.
    all_detections : Per-frame list of {frame, track_id, class, confidence}.
    """
    model = _get_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # imageio-ffmpeg writes H.264 MP4 — plays natively in every modern browser
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        output_params=["-preset", "fast", "-crf", "23", "-movflags", "+faststart"],
        macro_block_size=None,
    )

    trail_history: dict[int, deque] = {}
    team_tracker  = TeamTracker()               # stable team assignments across frames
    all_detections: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                conf=conf_threshold,
                tracker=config.TRACKER_CONFIG,
                persist=True,
                verbose=False,
            )[0]

            annotated = results.plot()

            # Collect per-frame detections and track IDs
            frame_dets: list[dict] = []
            frame_tids: list[int]  = []
            frame_kps:  list       = []   # per-detection COCO keypoints (or None)

            for i, box in enumerate(results.boxes):
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                det = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class":      results.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                }
                frame_dets.append(det)
                frame_tids.append(track_id)

                # Extract keypoints aligned to this box index
                kps: list | None = None
                if results.keypoints is not None and i < len(results.keypoints):
                    kpt     = results.keypoints[i]
                    kp_xy   = kpt.xy[0].tolist()
                    kp_conf = kpt.conf[0].tolist() if kpt.conf is not None else [1.0] * len(kp_xy)
                    kps = [
                        {
                            "name":       KEYPOINT_NAMES[j] if j < len(KEYPOINT_NAMES) else f"kp{j}",
                            "x":          float(kp_xy[j][0]),
                            "y":          float(kp_xy[j][1]),
                            "confidence": float(kp_conf[j]),
                        }
                        for j in range(len(kp_xy))
                    ]
                frame_kps.append(kps)

                if show_trails or show_trajectory:
                    update_trail(trail_history, track_id, cx, cy, max_len=trail_length)

                all_detections.append(
                    {"frame": frame_idx, "track_id": track_id,
                     "class": det["class"], "confidence": det["confidence"]}
                )

            # ── Team overlay ──────────────────────────────────────────────────
            if len(frame_dets) >= 2:
                assignments = team_tracker.update(frame, frame_dets, frame_tids, frame_kps)
                frame_labels = [assignments.get(tid, 0) for tid in frame_tids]
                draw_team_overlay(annotated, frame_dets, frame_labels,
                                  team_tracker.cluster_centers)

            # ── Trails + trajectory (drawn last, on top) ──────────────────────
            if show_trails:
                draw_trails(annotated, trail_history)
            if show_trajectory:
                draw_trajectories(annotated, trail_history, height, width)

            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        writer.close()

    return output_path, all_detections


def summarise_video_detections(all_detections: list[dict]) -> dict:
    """Aggregate per-frame detections into video-level stats."""
    class_counts: dict[str, int] = {}
    track_ids_per_class: dict[str, set] = {}

    for d in all_detections:
        cls = d["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        track_ids_per_class.setdefault(cls, set())
        if d["track_id"] != -1:
            track_ids_per_class[cls].add(d["track_id"])

    return {
        "total_detections":        len(all_detections),
        "class_counts":            class_counts,
        "unique_tracks_per_class": {
            cls: len(ids) for cls, ids in track_ids_per_class.items()
        },
    }
