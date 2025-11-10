import cv2
import mediapipe as mp
import time
import math
import subprocess
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurable parameters
CAM_WIDTH, CAM_HEIGHT = 1280, 720
SMOOTHING = 7  # smoothing for mouse
SWIPE_THRESHOLD = 120  # pixels for swipe slide change
SWIPE_TIME = 0.6  # seconds cooldown for swipe actions
CLICK_DIST_THRESHOLD = 30  # pixels threshold between index and middle finger for click
VOLUME_DISTANCE_MIN = 15
VOLUME_DISTANCE_MAX = 220

# State variables
prev_mouse_x, prev_mouse_y = 0, 0
mouse_smooth_x, mouse_smooth_y = 0, 0
last_swipe_time = 0
last_play_pause_time = 0
last_click_time = 0
last_volume_set = None
last_brightness_set = None

# Utility functions for system actions (macOS focused for volume)
def set_volume_mac(percent: int):
    percent = max(0, min(100, int(percent)))
    try:
        subprocess.run(["osascript", "-e", f"set volume output volume {percent}"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def set_brightness_percent(percent: int):
    # Requires 'brightness' command-line tool on macOS (https://github.com/nriley/brightness)
    percent = max(0, min(100, int(percent)))
    try:
        # brightness expects 0..1
        val = percent / 100.0
        subprocess.run(["brightness", str(val)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def press_space():
    try:
        pyautogui.press("space")
    except Exception:
        pass

def press_left():
    try:
        pyautogui.press("left")
    except Exception:
        pass

def press_right():
    try:
        pyautogui.press("right")
    except Exception:
        pass

def media_play_pause():
    # Try space (works for many web players) as a fallback
    press_space()

def left_click():
    try:
        pyautogui.click()
    except Exception:
        pass

def move_mouse_to(x, y):
    try:
        xi = int(max(0, min(pyautogui.size().width - 1, round(x))))
        yi = int(max(0, min(pyautogui.size().height - 1, round(y))))
        pyautogui.moveTo(xi, yi, duration=0.01)
    except Exception:
        pass

# Helper: determine which fingers are up (thumb, index, middle, ring, pinky)
def fingers_up(lm_list):
    # lm_list: list of 21 landmarks where each item is (id, x, y)
    if not lm_list or len(lm_list) < 21:
        return [False] * 5

    fingers = []
    # Tip and pip indices
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    # Thumb: use x comparison between tip and ip/mcp to determine openness (works for mirrored webcam)
    try:
        thumb_tip_x = lm_list[4][1]
        thumb_ip_x = lm_list[3][1]
        fingers.append(thumb_tip_x < thumb_ip_x)
    except Exception:
        fingers.append(False)

    # Other fingers: tip y < pip y -> finger up (y increases downward)
    for tid, pid in zip(tip_ids[1:], pip_ids[1:]):
        try:
            tip_y = lm_list[tid][2]
            pip_y = lm_list[pid][2]
            fingers.append(tip_y < pip_y)
        except Exception:
            fingers.append(False)

    return fingers  # [thumb, index, middle, ring, pinky]

# Distance between two landmarks
def distance_points(a, b):
    if not a or not b:
        return float('inf')
    return math.hypot(a[1] - b[1], a[2] - b[2])

def map_range(value, in_min, in_max, out_min, out_max):
    # Clamp first
    if value < in_min:
        value = in_min
    if value > in_max:
        value = in_max
    in_span = in_max - in_min
    out_span = out_max - out_min
    scaled = float(value - in_min) / float(in_span) if in_span != 0 else 0
    return out_min + (scaled * out_span)

def main():
    global prev_mouse_x, prev_mouse_y, mouse_smooth_x, mouse_smooth_y
    global last_swipe_time, last_play_pause_time, last_click_time
    global last_volume_set, last_brightness_set

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    screen_w, screen_h = pyautogui.size()

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        prev_index_x = None

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape
            lm_list = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ensure we have full set of landmarks
                if len(lm_list) < 21:
                    # skip this frame if incomplete
                    cv2.imshow("Hand Gesture Control", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # Compute finger states
                fingers = fingers_up(lm_list)  # [thumb, index, middle, ring, pinky]
                index_tip = lm_list[8]
                thumb_tip = lm_list[4]
                middle_tip = lm_list[12]

                # --- Volume control: thumb + index up, others down, use distance to set volume ---
                if fingers[0] and fingers[1] and not any(fingers[2:]):
                    dist = distance_points(thumb_tip, index_tip)
                    vol = int(map_range(dist, VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX, 0, 100))
                    # Avoid setting too frequently
                    if last_volume_set is None or abs(vol - last_volume_set) >= 2:
                        set_volume_mac(vol)
                        last_volume_set = vol
                    # Visual feedback
                    cv2.putText(frame, f"Volume: {vol}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # --- Brightness control: use thumb + middle up and others down (optional) ---
                elif fingers[0] and fingers[2] and not any([fingers[1], fingers[3], fingers[4]]):
                    dist_b = distance_points(thumb_tip, middle_tip)
                    bri = int(map_range(dist_b, VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX, 0, 100))
                    if last_brightness_set is None or abs(bri - last_brightness_set) >= 2:
                        set_brightness_percent(bri)
                        last_brightness_set = bri
                    cv2.putText(frame, f"Brightness: {bri}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,200,0), 2)

                # --- Virtual mouse: index up, others down (control mouse by index tip position) ---
                elif fingers[1] and not fingers[2]:
                    ix, iy = index_tip[1], index_tip[2]
                    # Map camera coords to screen coords and clamp
                    screen_x = np.interp(ix, (0, w), (0, screen_w - 1))
                    screen_y = np.interp(iy, (0, h), (0, screen_h - 1))
                    # smoothing
                    mouse_smooth_x = prev_mouse_x + (screen_x - prev_mouse_x) / SMOOTHING
                    mouse_smooth_y = prev_mouse_y + (screen_y - prev_mouse_y) / SMOOTHING
                    move_mouse_to(mouse_smooth_x, mouse_smooth_y)
                    prev_mouse_x, prev_mouse_y = mouse_smooth_x, mouse_smooth_y
                    cv2.circle(frame, (ix, iy), 10, (255, 0, 255), cv2.FILLED)
                    cv2.putText(frame, f"Mouse mode", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

                    # Click: if index and middle tips are close together, perform a click
                    # (User can bring middle finger close to index to click)
                    if distance_points(index_tip, middle_tip) < CLICK_DIST_THRESHOLD:
                        now = time.time()
                        if now - last_click_time > 0.6:
                            left_click()
                            last_click_time = now
                            cv2.putText(frame, "Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                else:
                    # --- Play/Pause: all fingers up and relatively stationary -> toggle ---
                    if all(fingers):
                        now = time.time()
                        ix = index_tip[1]
                        if prev_index_x is not None and now - last_play_pause_time > 1.0:
                            if abs(ix - prev_index_x) < 10:
                                media_play_pause()
                                last_play_pause_time = now
                                cv2.putText(frame, "Play/Pause", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # --- Slide change by horizontal swipe of index finger ---
                current_time = time.time()
                ix_now = index_tip[1]
                if prev_index_x is not None and (current_time - last_swipe_time > SWIPE_TIME):
                    dx = ix_now - prev_index_x
                    if dx > SWIPE_THRESHOLD:
                        # moved right
                        press_right()
                        last_swipe_time = current_time
                        cv2.putText(frame, "Slide ->", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)
                    elif dx < -SWIPE_THRESHOLD:
                        press_left()
                        last_swipe_time = current_time
                        cv2.putText(frame, "Slide <-", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,0), 2)

                prev_index_x = ix_now

            # Display frame
            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()