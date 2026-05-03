from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8s.pt")

input_video = "../input_video/football.mp4"
output_video = "../output/final_tracked_output.mp4"

cap = cv2.VideoCapture(input_video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Tracker ID remapping for clean labels
id_mapping = {}
next_clean_id = 1

results = model.track(
    source=input_video,
    tracker="botsort.yaml",
    conf=0.45,
    iou=0.5,
    persist=True,
    stream=True,
    classes=[0]
)

for result in results:
    frame = result.orig_img.copy()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)

            if track_id not in id_mapping:
                id_mapping[track_id] = next_clean_id
                next_clean_id += 1

            clean_id = id_mapping[track_id]
            label = f"Player {clean_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        cv2.putText(frame,
                    f"Tracked Players: {len(boxes)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    1)

    out.write(frame)

cap.release()
out.release()

print("Final Submission Output Generated Successfully!")