# Object Detection and Tracking

A beginner-friendly Streamlit app that detects objects in an uploaded video and follows them from frame to frame. Video processing, YOLO inference, and tracking run locally; the app does not use an external detection API or require API keys.

## Objectives

- Apply object detection to video frames with a pre-trained model.
- Use object tracking to associate detections across frames.
- Visualize object locations, labels, confidence, and tracking IDs.
- Summarize unique tracked objects by class.

## Features

- Upload MP4, MOV, AVI, MKV, or WebM videos.
- Optionally upload your own Ultralytics YOLO `.pt` weights file from your computer.
- Adjust the detection confidence threshold.
- Choose how often YOLO runs; processing every 2nd frame is the default for faster results.
- Draw bounding boxes, class names, confidence scores, and ByteTrack IDs.
- View the processed video and per-class unique track counts.
- Show helpful errors for unreadable videos and processing failures.

## Technologies

- Python and Streamlit for the user interface.
- OpenCV for video reading, frame annotation, and video writing.
- Ultralytics YOLO for local object detection and tracking.
- ByteTrack, supplied through Ultralytics, to associate detections across frames.
- NumPy and pandas for frame data and count summaries.

## YOLO and ByteTrack

YOLO is a family of object-detection models. By default, this app uses the small `yolo11n.pt` nano model, which predicts bounding boxes, class labels, and confidence scores. Ultralytics downloads its model weights the first time the app runs; no API key is needed. You can instead select **Upload a custom YOLO model** in the app and choose trained Ultralytics `.pt` weights on your computer. The selected weights are used for that processing run; they are not added permanently to the project.

ByteTrack associates detections between adjacent frames so an object can keep a tracking ID as it moves. The app calls Ultralytics tracking with `bytetrack.yaml` and persistent tracking enabled. IDs are unique within one processing run, not across separate uploads.

## Installation

Use Python 3.9 or newer. From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## How to run

```bash
streamlit run app.py
```

Streamlit prints a local address to open in your browser.

## How to upload a video

1. Open the app in your browser.
2. Optionally choose trained Ultralytics `.pt` weights from your computer, or leave the custom model field empty to use the default model.
3. Select a video with the upload control.
4. Adjust the confidence threshold and inference frequency if needed. Processing every frame is more accurate but slower.
5. Select **Start processing** and wait for the progress indicator.
6. View the annotated video and unique track counts by class.

## Project structure

```text
.
├── app.py            # Streamlit interface and video processing loop
├── detector.py       # YOLO model loading and detection formatting
├── tracker.py        # Persistent ByteTrack wrapper
├── utils.py          # Frame annotation and counting helpers
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

## CPU and Chromebook considerations

The nano model is smaller than larger YOLO models, but inference can still be slow on a CPU. Use short clips, modest resolutions, and a higher confidence threshold to reduce processing time. The first run downloads the model weights and therefore needs an internet connection; subsequent inference is local. OpenCV also needs a video codec available to write the processed MP4. Performance depends on the Chromebook's Linux/Python environment and available memory.