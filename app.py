"""Streamlit interface for local video object detection and tracking."""

import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from detector import ObjectDetector
from tracker import ObjectTracker
from utils import process_frame


st.set_page_config(page_title="Object Detection & Tracking", page_icon="🎥", layout="wide")
st.title("Object Detection & Tracking")
st.write("Detect objects and follow them across video frames, locally on your device.")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="For best performance on a CPU, use a short, low-resolution video.",
)
uploaded_model = st.file_uploader(
    "Upload a custom YOLO model (optional)",
    type=["pt"],
    help="Choose a trained Ultralytics YOLO .pt weights file from your computer. "
    "Leave empty to use the default yolo11n.pt model.",
)
confidence = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Detections below this confidence are ignored.",
)
frame_stride = st.select_slider(
    "Inference frequency",
    options=[1, 2, 3, 5],
    value=2,
    format_func=lambda stride: "Every frame" if stride == 1 else f"Every {stride} frames",
    help="Run YOLO on fewer frames to finish faster. Larger gaps may reduce tracking accuracy.",
)

if uploaded_video is not None and st.button("Start processing", type="primary"):
    with tempfile.TemporaryDirectory(prefix="object-tracking-") as temp_dir:
        input_path = Path(temp_dir) / f"input.{Path(uploaded_video.name).suffix.lstrip('.') or 'mp4'}"
        output_path = Path(temp_dir) / "processed.mp4"
        input_path.write_bytes(uploaded_video.getvalue())
        model_path = None
        if uploaded_model is not None:
            model_path = Path(temp_dir) / "custom_model.pt"
            model_path.write_bytes(uploaded_model.getvalue())

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            st.error("This video could not be opened. Try a different file or video format.")
            capture.release()
        else:
            success, first_frame = capture.read()
            if not success or first_frame is None:
                st.error("The uploaded file contains no readable video frames.")
                capture.release()
            else:
                frame_height, frame_width = first_frame.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 20.0
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / frame_stride,
                    (frame_width, frame_height),
                )

                if not writer.isOpened():
                    st.error("Could not create the processed video. Check that OpenCV video codecs are available.")
                    capture.release()
                    writer.release()
                else:
                    progress = st.progress(0, text="Loading the YOLO model...")
                    status = st.empty()
                    try:
                        detector = ObjectDetector(str(model_path) if model_path else "yolo11n.pt")
                        tracker = ObjectTracker(detector)
                        class_track_ids: dict[str, set[int]] = defaultdict(set)
                        processed_frames = 0
                        frame_index = 0
                        expected_processed_frames = (
                            (frame_count + frame_stride - 1) // frame_stride
                            if frame_count > 0
                            else 0
                        )
                        frame = first_frame

                        while True:
                            if frame_index % frame_stride == 0:
                                annotated, detections = process_frame(frame, tracker, confidence)
                                writer.write(annotated)
                                for detection in detections:
                                    track_id = detection["track_id"]
                                    if track_id is not None:
                                        class_track_ids[detection["class_name"]].add(track_id)
                                processed_frames += 1

                                if frame_count > 0 and (
                                    processed_frames % 5 == 0
                                    or processed_frames == expected_processed_frames
                                ):
                                    progress.progress(
                                        min(processed_frames / expected_processed_frames, 1.0),
                                        text=f"Processing sampled frame {processed_frames} of {expected_processed_frames}...",
                                    )
                                elif frame_count <= 0 and processed_frames % 10 == 0:
                                    status.info(f"Processed {processed_frames} sampled frames...")

                            success, frame = capture.read()
                            if not success:
                                break
                            frame_index += 1

                        writer.release()
                        capture.release()
                        progress.progress(
                            1.0,
                            text=f"Finished processing {processed_frames} sampled frames.",
                        )

                        if processed_frames == 0 or not output_path.is_file():
                            st.error("No video frames could be processed.")
                        else:
                            st.subheader("Processed video")
                            st.video(output_path.read_bytes())
                            st.subheader("Unique tracked objects")
                            st.metric("Total unique tracks", sum(map(len, class_track_ids.values())))
                            if class_track_ids:
                                summary = pd.DataFrame(
                                    [
                                        {"Class": name, "Unique tracks": len(track_ids)}
                                        for name, track_ids in sorted(class_track_ids.items())
                                    ]
                                )
                                st.dataframe(summary, hide_index=True, use_container_width=True)
                            else:
                                st.info("No tracked objects were found. Try lowering the confidence threshold.")
                    except Exception as error:
                        st.error(f"Video processing failed: {error}")
                    finally:
                        writer.release()
                        capture.release()
else:
    st.caption("Choose a video and confidence threshold, then start processing.")