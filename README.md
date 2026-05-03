# Multi-Object Detection and Persistent ID Tracking in Public Sports Footage

## Project Overview
This project implements a computer vision pipeline for detecting and persistently tracking multiple moving players in publicly available football footage. The objective is to identify all relevant human subjects frame-by-frame, assign unique IDs, and maintain those IDs as consistently as possible across the processed video segment.

The final system produces an annotated output video containing:
- Bounding boxes around each detected player
- Unique player IDs
- Center movement points
- Total tracked player count per frame

---

## Public Video Source
Original Public Video Link: https://youtu.be/M9mKnmt0YaM?si=MmPbVWQH2FzXQRME

Processed Segment Used: **00:00:45 to 00:00:58**

Reason for Segment Selection:
A continuous tactical football sequence with multiple visible players and relatively stable camera motion was selected to enable reliable multi-object detection and persistent ID assignment.

---

## Methodology / Pipeline Design

The implemented pipeline follows these stages:

### 1. Input Video Acquisition
A publicly accessible football broadcast video was selected from YouTube and a representative 13-second continuous segment was extracted.

### 2. Frame-by-Frame Processing
The video is processed sequentially frame-by-frame using OpenCV.

### 3. Object Detection
A pretrained YOLOv8s object detection model is used to detect human subjects (person class) in every frame.

### 4. Multi-Object Tracking
Detected player bounding boxes are passed into the BoT-SORT tracker, which assigns persistent tracking IDs and attempts to maintain identity continuity using motion association and appearance-based matching.

### 5. Clean ID Mapping
Internal tracker IDs are remapped into cleaner labels such as:
- Player 1
- Player 2
- Player 3

for improved readability.

### 6. Visualization
For each tracked player:
- bounding rectangle is drawn,
- unique player label is displayed,
- center position point is marked.

Additionally, the top-left corner displays the total number of tracked players visible in the current frame.

### 7. Output Video Generation
All annotated frames are reassembled and exported as the final processed MP4 video.

---

## Model and Tracker Used

### Detection Model:
YOLOv8s (Ultralytics)

Reason:
- fast inference
- strong person detection capability
- lightweight enough for local execution

### Tracking Algorithm:
BoT-SORT

Reason:
- better identity preservation than motion-only trackers
- combines Kalman prediction with appearance-assisted association
- suitable for sports multi-player footage

---

## Dependencies / Requirements

Install required libraries:

pip install ultralytics  
pip install opencv-python  
pip install numpy

---

## How to Run

Place the trimmed input football video inside:

input_video/football.mp4

Run the script from src folder:

python detector_tracker.py

The final annotated video will be generated at:

output/final_tracked_output.mp4

---

## Project Structure

multi_object_tracking_assignment/  
│  
├── input_video/  
│   └── football.mp4  
│  
├── output/  
│   └── final_tracked_output.mp4  
│  
├── src/  
│   └── detector_tracker.py  
│  
└── README.md

---

## Assumptions Taken

- Only human players/referee are considered as relevant tracking subjects.
- The selected video segment contains continuous gameplay with no abrupt scene cut.
- Publicly broadcast sports footage is sufficient for demonstrating persistent multi-object tracking.

---

## Limitations Observed

Although BoT-SORT improves identity continuity, occasional ID reassignment may still occur under:
- dense player clustering,
- fast camera panning,
- partial player occlusion,
- similar jersey appearance.

These are standard practical limitations in real-world multi-object tracking systems.

---

## Optional Enhancements Implemented

- Center movement point visualization
- Live tracked player count display
- Clean human-readable ID remapping

---

## Future Improvements

Possible improvements include:
- jersey color based team clustering,
- trajectory path visualization,
- player speed estimation,
- stronger ReID embedding model,
- top-view positional mapping.

---

## Conclusion

This project successfully demonstrates a practical end-to-end multi-object detection and persistent ID tracking pipeline on public football footage using YOLOv8s + BoT-SORT. The system detects multiple players, assigns track identities, and generates an annotated output video while handling real-world sports broadcast challenges reasonably effectively.