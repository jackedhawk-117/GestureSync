import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import time
import math
import numpy as np

# --- Profile Configuration ---
PROFILES = {
    "The Witcher 3": {
        "type": "named",
        "gestures": [
            {"name": "Roach", "key": "x", "detector": "is_whistle_gesture"},
            {"name": "Quen", "key": "5", "detector": "is_quen_gesture"},
            {"name": "Aard", "key": "3", "detector": "is_aard_gesture"},
            {"name": "Igni", "key": "4", "detector": "is_igni_gesture"},
            {"name": "Axii", "key": "6", "detector": "is_axii_gesture"},
        ],
        "color": (200, 0, 0)
    },
    "Ready or Not": {
        "type": "counting",
        "detector": "count_extended_fingers",
        "gestures": [
            {"name": "Open Commands", "key": Button.middle, "count": 0}, # New gesture for closed fist
            {"name": "Command 1", "key": "1", "count": 1},
            {"name": "Command 2", "key": "2", "count": 2},
            {"name": "Command 3", "key": "3", "count": 3},
            {"name": "Command 4", "key": "4", "count": 4},
            {"name": "Command 5", "key": "5", "count": 5},
        ],
        "color": (0, 100, 200)
    }
}

# --- General Configuration ---
CAM_WIDTH = 1280
CAM_HEIGHT = 720
GESTURE_COOLDOWN_FRAMES = 4
ACTION_COOLDOWN_SECONDS = 0.8

# --- Initialization ---
print("Initializing gesture controller...")
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

keyboard = KeyboardController()
mouse = MouseController()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# --- UI and State Variables ---
profile_keys = list(PROFILES.keys())
current_profile_index = 0
active_gesture = None
last_action_time = 0
gesture_counters = {}
dropdown_open = False
dropdown_rect = (50, 50, 300, 50) # x, y, width, height

# --- Mouse Click Handler ---
def handle_mouse_click(event, x, y, flags, param):
    global dropdown_open, current_profile_index, gesture_counters
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on the main dropdown box
        dx, dy, dw, dh = dropdown_rect
        if dx < x < dx + dw and dy < y < dy + dh:
            dropdown_open = not dropdown_open
            return

        # If dropdown is open, check for clicks on options
        if dropdown_open:
            for i, profile_name in enumerate(profile_keys):
                option_y = dy + dh * (i + 1)
                if dx < x < dx + dw and option_y < y < option_y + dh:
                    if current_profile_index != i:
                        current_profile_index = i
                        gesture_counters.clear() # Reset counters on profile switch
                        print(f"\nSwitched to {profile_keys[i]} Profile")
                    dropdown_open = False
                    return
        
        # If clicked outside, close dropdown
        dropdown_open = False

# --- Gesture Detection Functions ---
def get_landmark_coords(landmarks, landmark_index):
    lm = landmarks.landmark[landmark_index]
    return lm.x, lm.y

def is_finger_extended(h_lm, tip, pip):
    if not h_lm: return False
    _, tip_y = get_landmark_coords(h_lm, tip)
    _, pip_y = get_landmark_coords(h_lm, pip)
    return tip_y < pip_y

def is_aard_gesture(h_lm, f_lm):
    try:
        return (is_finger_extended(h_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                is_finger_extended(h_lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP))
    except: return False

def is_igni_gesture(h_lm, f_lm):
    try:
        thumb_tip_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_TIP)
        thumb_ip_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_IP)
        return (thumb_tip_x < thumb_ip_x and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP))
    except: return False

def is_quen_gesture(h_lm, f_lm):
    try:
        thumb_tip_x, thumb_tip_y = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_TIP)
        index_tip_x, index_tip_y = get_landmark_coords(h_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        distance = math.hypot(thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y)
        return (distance < 0.07 and
                is_finger_extended(h_lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                is_finger_extended(h_lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
                is_finger_extended(h_lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP))
    except: return False

def is_axii_gesture(h_lm, f_lm):
    try:
        thumb_tip_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_TIP)
        thumb_ip_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_IP)
        return (thumb_tip_x < thumb_ip_x and
                is_finger_extended(h_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP) and
                not is_finger_extended(h_lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP))
    except: return False

def is_whistle_gesture(h_lm, f_lm):
    if not (h_lm and f_lm): return False
    try:
        index_tip_x, index_tip_y = get_landmark_coords(h_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        lip_upper_x, lip_upper_y = get_landmark_coords(f_lm, 13)
        lip_lower_x, lip_lower_y = get_landmark_coords(f_lm, 14)
        mouth_center_x = (lip_upper_x + lip_lower_x) / 2
        mouth_center_y = (lip_upper_y + lip_lower_y) / 2
        distance = math.hypot(index_tip_x - mouth_center_x, index_tip_y - mouth_center_y)
        return distance < 0.05
    except: return False

def count_extended_fingers(h_lm, f_lm):
    if not h_lm: return -1 # Return an invalid count if no hand is detected
    count = 0
    try:
        # Check Thumb (logic is different for left vs right hand, this is for right)
        thumb_tip_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_TIP)
        thumb_mcp_x, _ = get_landmark_coords(h_lm, mp_hands.HandLandmark.THUMB_MCP)
        if thumb_tip_x < thumb_mcp_x:
            count += 1
        
        # Check other four fingers
        finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
        finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
        for i in range(4):
            if is_finger_extended(h_lm, finger_tips[i], finger_pips[i]):
                count += 1
    except: return -1
    return count

# --- Main Action Function ---
def perform_action(key, name):
    print(f"--- {name.upper()}! Action: Activating '{name}' ---")
    if isinstance(key, Button):
        mouse.press(key)
        time.sleep(0.1)
        mouse.release(key)
    else:
        keyboard.press(key)
        time.sleep(0.1)
        keyboard.release(key)
    print("--------------------------------------")

# --- UI Drawing Function ---
def draw_ui(image, profile_name, profile_color, status_text):
    # Draw dropdown menu
    dx, dy, dw, dh = dropdown_rect
    # Create a semi-transparent overlay for the menu
    overlay = image.copy()
    cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (40, 40, 40), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw text for current profile
    cv2.putText(image, f"Profile: {profile_name}", (dx + 10, dy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Draw dropdown arrow
    arrow_points = np.array([[(dx + dw - 30), dy + 20], [(dx + dw - 20), dy + 20], [(dx + dw - 25), dy + 30]], np.int32)
    cv2.fillPoly(image, [arrow_points], (255, 255, 255))

    if dropdown_open:
        for i, p_name in enumerate(profile_keys):
            option_y = dy + dh * (i + 1)
            cv2.rectangle(overlay, (dx, option_y), (dx + dw, option_y + dh), (60, 60, 60), -1)
            cv2.putText(overlay, p_name, (dx + 10, option_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw Gesture Status
    cv2.putText(image, f"Status: {status_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Draw Cooldown
    if (time.time() - last_action_time) <= ACTION_COOLDOWN_SECONDS:
        cv2.putText(image, "COOLDOWN", (CAM_WIDTH - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

# --- Main Loop ---
window_name = 'Multi-Profile Gesture Controller'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, handle_mouse_click)

print("Controller active. Use the dropdown in the GUI to switch profiles.")
print("Press 'q' in the preview window to quit.")

while cap.isOpened():
    profile_name = profile_keys[current_profile_index]
    profile = PROFILES[profile_name]
    if not gesture_counters:
        for gesture in profile["gestures"]:
            gesture_counters[gesture["name"]] = 0

    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb_image)
    face_results = face_mesh.process(rgb_image)
    
    hand_landmarks = hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None
    face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
    
    current_gesture_in_frame = None
    
    # --- Gesture Detection ---
    if hand_landmarks:
         mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if profile["type"] == "counting":
        finger_count = count_extended_fingers(hand_landmarks, face_landmarks)
        if finger_count >= 0: # Check now includes 0 for a closed fist
            matched_gesture = next((g for g in profile["gestures"] if g["count"] == finger_count), None)
            if matched_gesture:
                current_gesture_in_frame = matched_gesture["name"]
    elif profile["type"] == "named":
        for gesture in profile["gestures"]:
            detector_func = globals()[gesture["detector"]]
            if detector_func(hand_landmarks, face_landmarks):
                current_gesture_in_frame = gesture["name"]
                break

    # --- Action Logic ---
    if current_gesture_in_frame:
        if current_gesture_in_frame == active_gesture:
            gesture_counters[active_gesture] += 1
        else:
            active_gesture = current_gesture_in_frame
            for g_name in gesture_counters: gesture_counters[g_name] = 0
            if active_gesture in gesture_counters:
                gesture_counters[active_gesture] = 1
    else:
        active_gesture = None
        for g_name in gesture_counters: gesture_counters[g_name] = 0

    if active_gesture and (time.time() - last_action_time) > ACTION_COOLDOWN_SECONDS:
        if gesture_counters.get(active_gesture, 0) >= GESTURE_COOLDOWN_FRAMES:
            action_key = None
            if profile["type"] == "counting":
                finger_count = count_extended_fingers(hand_landmarks, face_landmarks)
                matched_gesture = next((g for g in profile["gestures"] if g["count"] == finger_count), None)
                if matched_gesture: action_key = matched_gesture["key"]
            elif profile["type"] == "named":
                action_key = next((g["key"] for g in profile["gestures"] if g["name"] == active_gesture), None)
            if action_key:
                perform_action(action_key, active_gesture)
                last_action_time = time.time()
                active_gesture = None
                for g_name in gesture_counters: gesture_counters[g_name] = 0

    # --- Draw UI and Display ---
    status_text = active_gesture if active_gesture else "No Gesture"
    image = draw_ui(image, profile_name, profile["color"], status_text)
    cv2.imshow(window_name, image)

    # --- Key Press Handling ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- Cleanup ---
print("Shutting down controller.")
cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
