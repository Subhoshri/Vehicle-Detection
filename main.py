import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
video_path = "/kaggle/input/vehant-vehicle-camera-dataset/Dataset/vehant_hackathon_video_1.avi"

BASE_MAX_DISTANCE = 50
NEAR_CAMERA_BOOST = 1.5
MAX_MISSING = 10

MIN_AGE = 8
MIN_MOVE = 15

COUNT_LINE_Y_RATIO = 0.6
VIS_EVERY_N_FRAMES = 150
EXIT_ZONE_Y_RATIO = 0.2   # top 20% of frame

# =========================================================
# HELPERS
# =========================================================
def is_valid_vehicle(obj):
    if obj["age"] < MIN_AGE:
        return False
    if len(obj["history"]) < 2:
        return False

    x0, y0 = obj["history"][0]
    x1, y1 = obj["history"][-1]
    return np.hypot(x1 - x0, y1 - y0) >= MIN_MOVE


def crossed_line(obj, line_y):
    if len(obj["history"]) < 2:
        return False
    (_, prev_y) = obj["history"][-2]
    (_, curr_y) = obj["history"][-1]
    return prev_y > line_y and curr_y <= line_y


# =========================================================
# MAIN
# =========================================================
cap = cv2.VideoCapture(video_path)

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

tracked_objects = {}
next_id = 0
vehicle_count = 0
frame_count = 0
count_line_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- Background subtraction ----------
    fg_mask = back_sub.apply(frame)
    _, fg_only = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # ---------- Morphology ----------
    fg_opened = cv2.morphologyEx(fg_only, cv2.MORPH_OPEN, kernel_open)
    fg_clean = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, kernel_close)

    h, w = fg_clean.shape
    frame_area = h * w

    if count_line_y is None:
        count_line_y = int(COUNT_LINE_Y_RATIO * h)
    exit_zone_y = int(EXIT_ZONE_Y_RATIO * h)

    # ---------- ROI ----------
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(0.05*h):h, int(0.05*w):int(0.95*w)] = 255
    fg_roi = cv2.bitwise_and(fg_clean, fg_clean, mask=roi_mask)

    # ---------- Contours ----------
    contours, _ = cv2.findContours(
        fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    current_centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 0.20 * frame_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch > 0.7 * h:
            continue

        if cw / float(ch) < 0.3:
            continue

        cx = x + cw // 2
        cy = y + ch // 2
        current_centroids.append((cx, cy))

    # ---------- Tracking ----------
    updated_objects = {}

    for cx, cy in current_centroids:
        matched_id = None
        best_score = float("inf")

        for obj_id, obj in tracked_objects.items():
            ox, oy = obj["centroid"]

            # direction consistency
            if cy > oy + 10:
                continue

            dist = np.hypot(cx - ox, cy - oy)
            dist_thresh = BASE_MAX_DISTANCE
            if oy > 0.7 * h:
                dist_thresh *= NEAR_CAMERA_BOOST

            if dist > dist_thresh:
                continue

            score = dist - 0.1 * obj["age"]
            if score < best_score:
                best_score = score
                matched_id = obj_id

        if matched_id is not None:
            obj = tracked_objects[matched_id]
            obj["centroid"] = (cx, cy)
            obj["last_seen"] = frame_count
            obj["age"] += 1
            obj["history"].append((cx, cy))
            updated_objects[matched_id] = obj
        else:
            updated_objects[next_id] = {
                "id": next_id,
                "centroid": (cx, cy),
                "last_seen": frame_count,
                "age": 1,
                "counted": False,
                "history": [(cx, cy)]
            }
            next_id += 1

    for obj_id, obj in tracked_objects.items():
        if frame_count - obj["last_seen"] <= MAX_MISSING:
            updated_objects[obj_id] = obj

    tracked_objects = updated_objects

    # ---------- Counting ----------
    for obj in tracked_objects.values():
        if not is_valid_vehicle(obj):
            continue
        if not obj["counted"] and crossed_line(obj, count_line_y):
            vehicle_count += 1
            obj["counted"] = True
        if not obj["counted"]:
            # Normal line crossing
            if crossed_line(obj, count_line_y):
                vehicle_count += 1
                obj["counted"] = True

            # Exit-zone counting (for trucks & edge cases)
            elif obj["centroid"][1] < exit_zone_y:
                vehicle_count += 1
                obj["counted"] = True

    # ---------- Visualization ----------
    if frame_count % VIS_EVERY_N_FRAMES == 0:
        vis = frame.copy()

        # counting line
        cv2.line(vis, (0, count_line_y), (w, count_line_y), (255, 0, 0), 2)

        for obj in tracked_objects.values():
            if not is_valid_vehicle(obj):
                continue

            cx, cy = obj["centroid"]
            label = f"ID {obj['id']}"
            if obj["counted"]:
                label += " ✓"

            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                vis, label, (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2
            )

        plt.figure(figsize=(10,4))
        plt.title(f"Frame {frame_count} | Total Count: {vehicle_count}")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    frame_count += 1

cap.release()

print("\n==============================")
print("FINAL VEHICLE COUNT:", vehicle_count)
print("==============================")
