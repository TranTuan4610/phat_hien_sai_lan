import cv2
import numpy as np

VIDEO_PATH = "input.mp4"
LANE_POINTS_FILE = "lane_points.npy"

lanes = []           # danh sách các lane (mỗi lane là list điểm)
current_points = []  # điểm đang vẽ
frame = None

def click_event(event, x, y, flags, param):
    global frame, current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # chấm đỏ
        if len(current_points) > 1:
            cv2.polylines(frame, [np.array(current_points, np.int32)], False, (0, 255, 0), 2)
        cv2.imshow("Select Lane Points", frame)

def main():
    global frame, current_points, lanes

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được video!")
        return

    cv2.imshow("Select Lane Points", frame)
    cv2.setMouseCallback("Select Lane Points", click_event)

    print("👉 Click chuột để chọn điểm cho lane (≥ 3 điểm).")
    print("👉 Nhấn ENTER để kết thúc 1 lane.")
    print("👉 Vẽ ít nhất 2 lane.")
    print("👉 Nhấn S để lưu, Q để thoát không lưu.")

    while True:
        cv2.imshow("Select Lane Points", frame)
        key = cv2.waitKey(1) & 0xFF

        # ENTER -> kết thúc lane
        if key in [13, 10]:
            if len(current_points) >= 3:
                lanes.append(current_points.copy())
                print(f"✅ Đã thêm lane {len(lanes)} với {len(current_points)} điểm.")
            else:
                print("⚠️ Cần ít nhất 3 điểm cho 1 lane.")
            current_points = []

        # S -> lưu lanes vào file
        elif key == ord("s"):
            if len(lanes) >= 2:   # bắt buộc phải có ≥ 2 lane
                if len(current_points) >= 3:
                    lanes.append(current_points.copy())
                np.save(LANE_POINTS_FILE, np.array(lanes, dtype=object))
                print(f"💾 Đã lưu {len(lanes)} lane vào {LANE_POINTS_FILE}")
            else:
                print("⚠️ Bạn phải vẽ ít nhất 2 lane mới được lưu.")
            break

        # Q -> thoát không lưu
        elif key == ord("q"):
            print("❌ Thoát không lưu.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
