"""
app.py — Streamlit entry point for SportVision.

Pipeline:
  Image → yolo11n-pose.pt (skeleton + detection in ONE pass) → OCR → Groq commentary
  Video → yolo11n-pose.pt + ByteTrack + trails + trajectory projection
"""

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

import config
from detector import (
    detect,
    detect_video,
    summarise_detections,
    summarise_video_detections,
)
from ocr_reader import read_text
from commentary import generate_commentary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SportVision",
    page_icon="🏟️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏟️ SportVision")
st.subheader("AI-Powered Sports Analytics")
st.markdown(
    "Upload a sports image **or** video clip — skeleton pose estimation, "
    "player tracking, jersey OCR, and AI commentary."
)
st.divider()

# ── Mode tabs ─────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📷 Image Analysis", "🎬 Video Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded_img = st.file_uploader(
        "Upload a sports image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPG, JPEG, PNG, WEBP",
        key="img_uploader",
    )

    if uploaded_img is not None:
        suffix = Path(uploaded_img.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_img.read())
            tmp_path = tmp.name

        try:
            # Load BGR array for OCR before temp file is deleted
            original_bgr = cv2.imread(tmp_path)

            # ── Single model call: detection + skeleton + teams in one pass ────
            with st.spinner("🦴 Running YOLOv11-Pose + team clustering…"):
                annotated_bgr, detections, poses, team_info = detect(tmp_path)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # ── Side-by-side: original | annotated with skeleton + team boxes ─
            col_orig, col_ann = st.columns(2)

            with col_orig:
                st.subheader("📷 Original")
                uploaded_img.seek(0)
                st.image(uploaded_img, use_container_width=True)

            with col_ann:
                st.subheader("🦴 Pose · Teams  ·  yolo11n-pose.pt")
                n_kp = sum(p["keypoints_visible"] for p in poses)
                st.caption(f"{len(poses)} player(s) · {n_kp} keypoints · "
                           f"{'2 teams detected' if team_info else 'team clustering needs ≥2 players'}")
                st.image(annotated_rgb, use_container_width=True)

            # ── Detection stats ───────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Detection Stats")

            if detections:
                counts = summarise_detections(detections)
                metric_cols = st.columns(max(len(counts), 1))
                for col, (cls_name, cnt) in zip(metric_cols, counts.items()):
                    col.metric(label=cls_name.title(), value=cnt)

                st.markdown("**All detections** (sorted by confidence)")
                st.dataframe(
                    detections,
                    use_container_width=True,
                    column_order=["class", "confidence", "x1", "y1", "x2", "y2"],
                )
            else:
                st.warning("No persons detected. Try an image with visible players.")

            # ── Team breakdown ────────────────────────────────────────────────
            st.divider()
            st.subheader("👕 Team Separation  ·  K-Means on Jersey Colour")

            if team_info:
                centers = team_info["centers"]
                counts  = team_info["counts"]

                def bgr_to_hex(bgr) -> str:
                    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
                    return f"#{r:02x}{g:02x}{b:02x}"

                hex_a = bgr_to_hex(centers[0])
                hex_b = bgr_to_hex(centers[1])

                # Styled team cards showing count + detected jersey swatch
                st.markdown(
                    f"""
                    <div style="display:flex;gap:16px;margin:8px 0 16px;">
                      <div style="flex:1;background:#1e2130;border-left:6px solid #FFB428;
                                  border-radius:8px;padding:14px 18px;">
                        <div style="color:#FFB428;font-weight:700;font-size:1rem;">🟠 Team A</div>
                        <div style="color:#eee;font-size:1.8rem;font-weight:700;">{counts[0]}</div>
                        <div style="color:#aaa;font-size:0.85rem;">players</div>
                        <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
                          <div style="width:24px;height:24px;border-radius:4px;
                                      background:{hex_a};border:1px solid #555;"></div>
                          <span style="color:#ccc;font-size:0.8rem;">dominant jersey colour</span>
                        </div>
                      </div>
                      <div style="flex:1;background:#1e2130;border-left:6px solid #2850FF;
                                  border-radius:8px;padding:14px 18px;">
                        <div style="color:#6699FF;font-weight:700;font-size:1rem;">🔵 Team B</div>
                        <div style="color:#eee;font-size:1.8rem;font-weight:700;">{counts[1]}</div>
                        <div style="color:#aaa;font-size:0.85rem;">players</div>
                        <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
                          <div style="width:24px;height:24px;border-radius:4px;
                                      background:{hex_b};border:1px solid #555;"></div>
                          <span style="color:#ccc;font-size:0.8rem;">dominant jersey colour</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Per-player team table
                team_rows = [
                    {
                        "player":     f"Person {i+1}",
                        "team":       f"Team {'A' if lbl == 0 else 'B'}",
                        "confidence": detections[i]["confidence"],
                    }
                    for i, lbl in enumerate(team_info["labels"])
                ]
                st.dataframe(team_rows, use_container_width=True)
                st.caption("Boxes in the annotated image above are colour-coded to match team assignment.")
            else:
                st.info("Upload an image with **2 or more players** to enable team separation.")

            # ── Pose keypoints detail ─────────────────────────────────────────
            st.divider()
            st.subheader("🦴 Pose Keypoints Detail")

            if poses:
                # Summary row per person
                summary_rows = [
                    {
                        "person":            f"Person {p['person']}",
                        "bbox confidence":   p["confidence"],
                        "keypoints visible": f"{p['keypoints_visible']} / 17",
                    }
                    for p in poses
                ]
                st.dataframe(summary_rows, use_container_width=True)

                # Expandable per-person keypoint table
                for p in poses:
                    with st.expander(
                        f"Person {p['person']}  ·  "
                        f"{p['keypoints_visible']}/17 keypoints  ·  "
                        f"conf {p['confidence']:.0%}"
                    ):
                        st.dataframe(p["keypoints"], use_container_width=True)
            else:
                st.info("No pose data — upload an image containing visible people.")

            # ── OCR ───────────────────────────────────────────────────────────
            st.divider()
            st.subheader("🔤 OCR — Jersey Numbers & Scoreboard Text")

            with st.spinner("🔤 Running EasyOCR…"):
                ocr_results = read_text(original_bgr)

            if ocr_results:
                st.markdown("**Detected text** (sorted by confidence):")
                cols_per_row = 6
                for row_start in range(0, len(ocr_results), cols_per_row):
                    row_items = ocr_results[row_start : row_start + cols_per_row]
                    chip_cols = st.columns(len(row_items))
                    for chip_col, item in zip(chip_cols, row_items):
                        chip_col.metric(
                            label=f"conf {item['confidence']:.0%}",
                            value=item["text"],
                        )
                with st.expander("Raw OCR table"):
                    st.dataframe(ocr_results, use_container_width=True)
            else:
                st.info("No text detected.")

            # ── AI Commentary ─────────────────────────────────────────────────
            st.divider()
            st.subheader("🎙️ AI Commentary")
            st.caption("Powered by Groq · llama-3.3-70b-versatile — set GROQ_API_KEY first.")

            if st.button("✨ Generate Commentary", type="primary", key="img_commentary_btn"):
                try:
                    with st.spinner("🎙️ Calling Groq…"):
                        commentary = generate_commentary(detections, ocr_results)

                    st.markdown(
                        f"""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                            border-left:4px solid #e94560;border-radius:8px;
                            padding:20px 24px;margin-top:12px;color:#eaeaea;
                            font-size:1.1rem;line-height:1.7;font-style:italic;">
                            🎙️ {commentary}</div>""",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Copy raw text"):
                        st.code(commentary, language=None)

                except ValueError as e:
                    st.error(f"**API Key Missing** — {e}")
                except Exception as e:
                    st.error(f"**Groq API error:** {e}")

        finally:
            os.unlink(tmp_path)

    else:
        st.info("👆 Upload an image to get started.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown(
        "Upload a sports clip. Every frame gets **YOLOv11-Pose** skeleton detection "
        "with **ByteTrack** player IDs, **motion trails**, and **trajectory projection**."
    )
    st.warning("⏱️ Clips under **30 seconds** recommended for quick demos.")

    uploaded_vid = st.file_uploader(
        "Upload a sports video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
    )

    if uploaded_vid is not None:
        # ── Visual overlay controls ───────────────────────────────────────────
        st.markdown("**Visual overlays:**")
        ov1, ov2, ov3 = st.columns(3)
        show_trails     = ov1.checkbox("Motion trails",         value=True)
        show_trajectory = ov2.checkbox("Trajectory projection",  value=True)
        trail_length    = ov3.slider("Trail length (frames)",
                                     config.TRAIL_LEN_MIN, config.TRAIL_LEN_MAX,
                                     config.TRAIL_LEN_DEFAULT, step=config.TRAIL_LEN_STEP)

        # Save uploaded video to a temp input file
        in_suffix = Path(uploaded_vid.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix) as tmp_in:
            tmp_in.write(uploaded_vid.read())
            in_path = tmp_in.name

        # Separate temp file for H.264 output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            out_path = tmp_out.name

        try:
            progress_bar = st.progress(0, text="Starting…")
            status_text  = st.empty()

            def update_progress(current: int, total: int) -> None:
                pct = current / total
                progress_bar.progress(pct, text=f"Frame {current} / {total}")
                status_text.caption(f"{pct:.0%} complete")

            # Run pose detection + ByteTrack + trails + trajectory
            _, all_detections = detect_video(
                in_path,
                out_path,
                show_trails=show_trails,
                show_trajectory=show_trajectory,
                trail_length=trail_length,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="✅ Done!")
            status_text.empty()

            # ── Annotated video playback (H.264 — plays in browser) ───────────
            st.divider()
            st.subheader("🎬 Annotated Output")

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Download annotated video",
                data=video_bytes,
                file_name=f"sportvision_{Path(uploaded_vid.name).stem}_annotated.mp4",
                mime="video/mp4",
            )

            # ── Tracking stats ────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Tracking Stats")

            if all_detections:
                stats = summarise_video_detections(all_detections)
                cc, ut = stats["class_counts"], stats["unique_tracks_per_class"]

                m1, m2 = st.columns(2)
                m1.metric("Total detections (all frames)", stats["total_detections"])
                m2.metric("Unique player IDs (ByteTrack)", sum(ut.values()))

                st.dataframe(
                    [
                        {"class": cls, "total detections": cc[cls], "unique tracks": ut.get(cls, 0)}
                        for cls in cc
                    ],
                    use_container_width=True,
                )
            else:
                st.warning("No persons detected in the video.")

            # ── AI Commentary ─────────────────────────────────────────────────
            st.divider()
            st.subheader("🎙️ AI Commentary")

            if st.button("✨ Generate Commentary", type="primary", key="vid_commentary_btn"):
                if all_detections:
                    try:
                        with st.spinner("🎙️ Calling Groq…"):
                            seen: set[tuple] = set()
                            deduped = []
                            for d in all_detections:
                                key = (d["class"], d["track_id"])
                                if key not in seen:
                                    deduped.append({"class": d["class"], "confidence": d["confidence"],
                                                    "x1": 0, "y1": 0, "x2": 0, "y2": 0})
                                    seen.add(key)
                            commentary = generate_commentary(deduped, [])

                        st.markdown(
                            f"""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                                border-left:4px solid #e94560;border-radius:8px;
                                padding:20px 24px;margin-top:12px;color:#eaeaea;
                                font-size:1.1rem;line-height:1.7;font-style:italic;">
                                🎙️ {commentary}</div>""",
                            unsafe_allow_html=True,
                        )
                        with st.expander("Copy raw text"):
                            st.code(commentary, language=None)

                    except ValueError as e:
                        st.error(f"**API Key Missing** — {e}")
                    except Exception as e:
                        st.error(f"**Groq API error:** {e}")
                else:
                    st.warning("No detections available for commentary.")

        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    else:
        st.info("👆 Upload a video clip to get started.")
