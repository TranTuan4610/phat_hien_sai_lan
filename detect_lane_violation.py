import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output.mp4"
POINTS_FILE = "lane_points.npy"

# Load lane polygons
LANES = np.load(POINTS_FILE, allow_pickle=True)

# Fix: nếu chỉ chọn 1 polygon thì ép thành list
if LANES.ndim == 2:
    LANES = [LANES]

model = YOLO("yolov8n.pt")

# =========================
# 1. Utilities
# =========================
def point_in_poly(point, poly):
    """Kiểm tra 1 điểm có nằm trong polygon không"""
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1) * (y-y1) / (y2-y1+1e-9) + x1):
            inside = not inside
    return inside

def point_near_line(px, py, x1, y1, x2, y2, thresh=8):
    """Kiểm tra điểm (px,py) có gần đoạn thẳng (x1,y1)-(x2,y2) không"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx, yy = x1 + param * C, y1 + param * D

    dx = px - xx
    dy = py - yy
    return (dx * dx + dy * dy) ** 0.5 < thresh

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

# =========================
# 2. Simple Tracker (ID)
# =========================
class SimpleTracker:
    def __init__(self, max_dist=50):
        self.objects = {}   # id -> data
        self.next_id = 1
        self.max_dist = max_dist

    def update(self, detections):
        assigned = {}
        centroids = [centroid(b) for b in detections]
        used = set()

        # gán bbox vào object cũ nếu gần
        for obj_id, obj in list(self.objects.items()):
            best_i, best_d = -1, 1e9
            for i, c in enumerate(centroids):
                if i in used:
                    continue
                d = math.hypot(c[0]-obj["centroid"][0], c[1]-obj["centroid"][1])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i != -1 and best_d < self.max_dist:
                self.objects[obj_id]["bbox"] = detections[best_i]
                self.objects[obj_id]["centroid"] = centroids[best_i]
                used.add(best_i)
                assigned[obj_id] = detections[best_i]
            else:
                # nếu mất dấu thì xoá
                self.objects.pop(obj_id, None)

        # tạo object mới
        for i, bbox in enumerate(detections):
            if i in used:
                continue
            cid = self.next_id
            self.next_id += 1
            self.objects[cid] = {
                "bbox": bbox,
                "centroid": centroids[i],
                "lane_history": deque(maxlen=10),
                "violation": False
            }
            assigned[cid] = bbox

        return assigned

# =========================
# 3. Main
# =========================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    tracker = SimpleTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in [2,3,5,7]:  # car, motorbike, bus, truck
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append((x1,y1,x2,y2))

        tracked = tracker.update(detections)

        # xử lý từng xe
        for obj_id, obj in tracker.objects.items():
            x1,y1,x2,y2 = obj["bbox"]
            cx, cy = obj["centroid"]

            # check lane hiện tại + chạm vạch
            current_lane = None
            touching_line = False
            for i, poly in enumerate(LANES):
                if point_in_poly((cx, cy), poly):
                    current_lane = i

                # check nếu centroid chạm bất kỳ cạnh nào của lane
                for j in range(len(poly)):
                    x1l, y1l = poly[j]
                    x2l, y2l = poly[(j+1) % len(poly)]
                    if point_near_line(cx, cy, x1l, y1l, x2l, y2l, thresh=8):
                        touching_line = True
                        break
                if touching_line:
                    break

            # lưu vào history
            obj["lane_history"].append(current_lane)

            # check violation
            lanes_seen = [l for l in obj["lane_history"] if l is not None]
            if len(set(lanes_seen)) > 1 or touching_line:
                obj["violation"] = True

            # vẽ bbox
            color = (0,0,255) if obj["violation"] else (0,255,0)
            label = f"ID {obj_id}"
            if obj["violation"]:
                label += " VIOLATION"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (cx,cy), 3, color, -1)

        # vẽ lane (sau cùng để nổi bật)
        for i, poly in enumerate(LANES):
            cv2.polylines(frame, [np.array(poly, np.int32)], True, (0,255,255), 3)
            cv2.putText(frame, f"Lane {i+1}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,255), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Lane Violation", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Done! Output saved at:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
