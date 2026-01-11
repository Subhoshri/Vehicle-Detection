import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
VIDEO_PATH = "/kaggle/input/vehant-vehicle-camera-dataset/Dataset/vehant_hackathon_video_6.avi"
MIN_AREA_RATIO = 0.0012
MIN_ASPECT_RATIO = 0.7

# Base parameters tuned for 20 FPS (reference FPS)
REFERENCE_FPS = 20.0
BASE_MAX_LIFESPAN_MISS = 15  # frames at 20 FPS
BASE_MIN_AGE_TO_COUNT = 3    # frames at 20 FPS
BASE_SEARCH_DIST_RATIO = 0.12  # ratio of frame height

COUNT_LINE_Y_RATIO = 0.60
VIS_INTERVAL = 20   # visualization interval (in frames)

def infer_direction(history, min_frames=4):
    if len(history) < min_frames:
        return None

    dy = history[-1][1] - history[0][1]

    if abs(dy) < 5:   # too little movement
        return None

    return 1 if dy > 0 else -1

# =========================================================
# INITIALIZATION
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"

# Get actual FPS from video
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps == 0:
    actual_fps = REFERENCE_FPS  # fallback
    print(f"Warning: Could not detect FPS, using {REFERENCE_FPS} FPS")
else:
    print(f"Detected video FPS: {actual_fps}")

# Calculate FPS normalization factor
fps_factor = actual_fps / REFERENCE_FPS
print(f"FPS normalization factor: {fps_factor:.2f}")

# Normalize parameters based on FPS
MAX_LIFESPAN_MISS = max(1, int(BASE_MAX_LIFESPAN_MISS * fps_factor))
MIN_AGE_TO_COUNT = max(1, int(BASE_MIN_AGE_TO_COUNT * fps_factor))

print(f"Adjusted MAX_LIFESPAN_MISS: {MAX_LIFESPAN_MISS} frames")
print(f"Adjusted MIN_AGE_TO_COUNT: {MIN_AGE_TO_COUNT} frames")

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=int(500 * fps_factor), 
    varThreshold=50, 
    detectShadows=True
)

tracked_objects = {}
next_id = 0
vehicle_count = 0
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w
    count_line_y = int(h * COUNT_LINE_Y_RATIO)

    # Adaptive kernel
    k_w = max(3, int(w * 0.005))
    k_h = max(15, int(h * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))

    # -----------------------------------------------------
    # 1. Background Subtraction
    # -----------------------------------------------------
    l_rate = 1.0 if frame_idx == 0 else (0.005 / fps_factor)
    fg_mask = back_sub.apply(frame, learningRate=l_rate)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # -----------------------------------------------------
    # 2. Contours + Perspective filtering
    # -----------------------------------------------------
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    current_centroids = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        perspective_factor = 0.2 + (0.8 * (y / h))
        if area < (frame_area * MIN_AREA_RATIO * perspective_factor):
            continue
        if (cw / ch) < MIN_ASPECT_RATIO:
            continue

        current_centroids.append((x + cw // 2, y + ch // 2))

    # -----------------------------------------------------
    # 3. Tracking (FPS-normalized search distance)
    # -----------------------------------------------------
    new_tracked = {}
    # Scale search distance inversely with FPS (higher FPS = smaller inter-frame movement)
    dynamic_search_dist = int(h * BASE_SEARCH_DIST_RATIO / fps_factor)

    for cx, cy in current_centroids:
        best_id = None
        min_dist = dynamic_search_dist

        for obj_id, data in tracked_objects.items():
            px, py = data["history"][-1]
            dist = np.hypot(cx - px, cy - py)

            if dist < min_dist:
                min_dist = dist
                best_id = obj_id

        if best_id is not None:
            data = tracked_objects.pop(best_id)
            data["history"].append((cx, cy))
            data["age"] += 1
            data["lost"] = 0
            new_tracked[best_id] = data
        else:
            new_tracked[next_id] = {
                "history": [(cx, cy)],
                "age": 1,
                "counted": False,
                "lost": 0,
                "direction": None,   # +1 = down, -1 = up
            }
            next_id += 1

    # Carry over lost tracks
    for obj_id, data in tracked_objects.items():
        if data["lost"] < MAX_LIFESPAN_MISS:
            data["lost"] += 1
            new_tracked[obj_id] = data

    tracked_objects = new_tracked

    # -----------------------------------------------------
    # COUNTING (direction-agnostic, single count per object)
    # -----------------------------------------------------
    for data in tracked_objects.values():

        if data["counted"]:
            continue

        if data["age"] < MIN_AGE_TO_COUNT:
            continue

        if len(data["history"]) < 2:
            continue

        y_prev = data["history"][-2][1]
        y_curr = data["history"][-1][1]

        # direction-agnostic crossing
        if (y_prev - count_line_y) * (y_curr - count_line_y) < 0:
            vehicle_count += 1
            data["counted"] = True

    # -----------------------------------------------------
    # 5. VISUALIZATION (EVERY N FRAMES)
    # -----------------------------------------------------
    # if frame_idx % VIS_INTERVAL == 0:
    #     vis = frame.copy()
    
    #     # draw counting line
    #     cv2.line(vis, (0, count_line_y), (w, count_line_y), (0, 0, 255), 2)
    
    #     # draw tracks
    #     for obj_id, data in tracked_objects.items():
    #         cx, cy = data["history"][-1]
    #         cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
    #         cv2.putText(
    #             vis, f"ID {obj_id}",
    #             (cx - 10, cy - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 0), 2
    #         )
    
    #     cv2.putText(
    #         vis, f"TOTAL COUNT: {vehicle_count}",
    #         (20, 40),
    #         cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #         (0, 255, 255), 3
    #     )
    
    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Frame {frame_idx} | FPS: {actual_fps:.1f}")
    #     plt.axis("off")
    #     plt.show()

    frame_idx += 1

cap.release()
print(f"\n--- FINAL VEHICLE COUNT: {vehicle_count} ---")
print(f"Video FPS: {actual_fps:.1f} | Frames processed: {frame_idx}")
