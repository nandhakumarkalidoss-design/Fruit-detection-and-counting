
import cv2
import os
from pathlib import Path
from collections import defaultdict, Counter
from tkinter import Tk, filedialog, messagebox
from ultralytics import YOLO

# ==================== SETTINGS ====================
MODEL_PATH = Path(
    r"C:\Users\Lenovo\fruit\fruits\runs\fruit_training\fruit_fast_30\weights\best.pt"
)
CONFIDENCE = 0.20
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
MAX_DETECTIONS = 300
LINE_POSITION = 0.58
TRACKER = "bytetrack.yaml"
# ==================================================

def choose_video():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select fruit video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    return path

def main():
    if not MODEL_PATH.exists():
        messagebox.showerror(
            "Model not found",
            f"Check the trained model path:\n{MODEL_PATH}"
        )
        return

    video_path = choose_video()
    if not video_path:
        print("No video selected.")
        return

    print("Loading trained model...")
    model = YOLO(str(MODEL_PATH))
    names = model.names

    print("Fruit classes:", names)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        messagebox.showerror("Error", "Cannot read video frames.")
        return

    height, width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path = str(
        Path(video_path).with_name(
            Path(video_path).stem + "_fruit_counted.mp4"
        )
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        messagebox.showerror("Error", "Cannot create output video.")
        return

    # Per-class crossing counts
    counts = Counter()

    # Track state
    previous_centers = {}
    class_votes = defaultdict(Counter)
    counted_ids = set()
    line_y = int(height * LINE_POSITION)
    frame_number = 0

    print("Processing video. Press Q to stop, P to pause.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                max_det=MAX_DETECTIONS,
                verbose=False
            )

            result = results[0]
            current_ids = set()

            # Detection and tracking
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                coords = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = None

                for i, box in enumerate(coords):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(class_ids[i])
                    confidence = float(confidences[i])
                    class_name = str(names[class_id])

                    track_id = (
                        int(track_ids[i])
                        if track_ids is not None else None
                    )

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if track_id is not None:
                        current_ids.add(track_id)

                        # Vote on class over time to reduce label flicker.
                        class_votes[track_id][class_name] += 1
                        stable_class = class_votes[track_id].most_common(1)[0][0]

                        previous = previous_centers.get(track_id)

                        # Count the tracked fruit only once when it crosses.
                        if previous is not None and track_id not in counted_ids:
                            old_y = previous[1]
                            crossed_line = (
                                old_y < line_y <= center_y
                                or old_y > line_y >= center_y
                            )

                            if crossed_line:
                                counts[stable_class] += 1
                                counted_ids.add(track_id)
                                print(
                                    f"Counted {stable_class}, "
                                    f"track ID {track_id}"
                                )

                        previous_centers[track_id] = (
                            center_x, center_y
                        )
                    else:
                        stable_class = class_name

                    label = f"{stable_class} {confidence:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 220, 0), 2
                    )
                    cv2.putText(
                        frame, label,
                        (x1, max(22, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 220, 0), 2
                    )
                    cv2.circle(
                        frame, (center_x, center_y),
                        3, (0, 0, 255), -1
                    )

            # Draw the counting line
            cv2.line(
                frame, (0, line_y), (width, line_y),
                (0, 255, 255), 3
            )
            cv2.putText(
                frame, "COUNTING LINE",
                (10, max(25, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2
            )

            # Count detections in the current frame.
            current_counts = Counter()
            if result.boxes is not None and len(result.boxes) > 0:
                for class_id in result.boxes.cls.cpu().numpy().astype(int):
                    current_counts[str(names[class_id])] += 1

            # Display per-class counts.
            panel_height = min(height - 4, 65 + len(names) * 24)
            cv2.rectangle(
                frame, (5, 5), (330, panel_height),
                (20, 20, 20), -1
            )
            cv2.putText(
                frame, "FRUIT COUNTS",
                (13, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            y_text = 48
            for class_id in sorted(names.keys()):
                fruit = str(names[class_id])
                text = (
                    f"{fruit}: {counts[fruit]} "
                    f"(now {current_counts[fruit]})"
                )
                cv2.putText(
                    frame, text,
                    (13, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (0, 255, 0), 1
                )
                y_text += 24
                if y_text > height - 5:
                    break

            writer.write(frame)
            cv2.imshow("YOLOv8 All Fruit Detection and Counting", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                while True:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key in (ord("p"), ord("q"), 27):
                        break
                if pause_key in (ord("q"), 27):
                    break

    except Exception as error:
        print("Processing error:", error)
        messagebox.showerror("Processing Error", str(error))

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\nFinal line-crossing counts:")
    for class_id in sorted(names.keys()):
        fruit = str(names[class_id])
        print(f"{fruit}: {counts[fruit]}")

    print("\nOutput saved:", output_path)
    messagebox.showinfo(
        "Processing Complete",
        f"Output video saved:\n{output_path}\n\n"
        + "\n".join(
            f"{names[class_id]}: {counts[str(names[class_id])]}"
            for class_id in sorted(names.keys())
        )
    )

if __name__ == "__main__":
    main()
