import cv2
import numpy as np

class Solution:
    def __init__(self):
        # Reference FPS used for normalization
        self.REFERENCE_FPS = 20.0

        # Detection & filtering parameters
        self.MIN_AREA_RATIO = 0.0012
        self.MIN_ASPECT_RATIO = 0.7

        # Tracking parameters (reference @ 20 FPS)
        self.BASE_MAX_LIFESPAN_MISS = 15
        self.BASE_MIN_AGE_TO_COUNT = 3
        self.BASE_SEARCH_DIST_RATIO = 0.12

        # Counting line position
        self.COUNT_LINE_Y_RATIO = 0.60

    def forward(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        # Detect FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = self.REFERENCE_FPS

        fps_factor = actual_fps / self.REFERENCE_FPS

        MAX_LIFESPAN_MISS = max(1, int(self.BASE_MAX_LIFESPAN_MISS * fps_factor))
        MIN_AGE_TO_COUNT = max(1, int(self.BASE_MIN_AGE_TO_COUNT * fps_factor))

        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=int(500 * fps_factor),
            varThreshold=50,
            detectShadows=True
        )

        tracked_objects = {}
        next_id = 0
        vehicle_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            frame_area = h * w
            count_line_y = int(h * self.COUNT_LINE_Y_RATIO)

            # Adaptive morphology kernel
            k_w = max(3, int(w * 0.005))
            k_h = max(15, int(h * 0.03))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))

            # --- Background subtraction ---
            lr = 1.0 if frame_idx == 0 else (0.005 / fps_factor)
            fg_mask = back_sub.apply(frame, learningRate=lr)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # --- Contour extraction ---
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            current_centroids = []

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)

                perspective_factor = 0.2 + 0.8 * (y / h)
                if area < frame_area * self.MIN_AREA_RATIO * perspective_factor:
                    continue
                if (cw / ch) < self.MIN_ASPECT_RATIO:
                    continue

                current_centroids.append((x + cw // 2, y + ch // 2))

            # --- Tracking ---
            new_tracked = {}
            search_dist = int(h * self.BASE_SEARCH_DIST_RATIO / fps_factor)

            for cx, cy in current_centroids:
                best_id = None
                min_dist = search_dist

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
                        "lost": 0
                    }
                    next_id += 1

            for obj_id, data in tracked_objects.items():
                if data["lost"] < MAX_LIFESPAN_MISS:
                    data["lost"] += 1
                    new_tracked[obj_id] = data

            tracked_objects = new_tracked

            # --- Counting (direction-agnostic with hysteresis) ---
            LINE_TOL = max(3, int(0.005 * h))

            for data in tracked_objects.values():
                if data["counted"]:
                    continue
                if data["age"] < MIN_AGE_TO_COUNT:
                    continue
                if len(data["history"]) < 2:
                    continue

                y_prev = data["history"][-2][1]
                y_curr = data["history"][-1][1]

                prev_dist = y_prev - count_line_y
                curr_dist = y_curr - count_line_y

                crossed = (
                    (prev_dist > LINE_TOL and curr_dist < -LINE_TOL) or
                    (prev_dist < -LINE_TOL and curr_dist > LINE_TOL)
                )

                if crossed:
                    vehicle_count += 1
                    data["counted"] = True

            frame_idx += 1

        cap.release()
        return vehicle_count
