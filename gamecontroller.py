import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math

# --- Configuration ---
# Keys to press for signs in The Witcher 3.
# Change these if your keybindings are different.
AARD_KEY = '3'
IGNI_KEY = '4'
QUEN_KEY = '5'
AXII_KEY = '6' # New key for Axii

# Webcam resolution
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Gesture detection parameters
# How many consecutive frames a gesture must be detected for it to trigger.
GESTURE_COOLDOWN_FRAMES = 10 
# How long to wait (in seconds) after casting before detecting another gesture.
ACTION_COOLDOWN_SECONDS = 2.0

# --- Initialization ---
print("Initializing gesture controller...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize Keyboard Controller
keyboard = Controller()

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# --- State Variables ---
# Use a dictionary to keep track of counters for each gesture
gesture_counters = {"Aard": 0, "Igni": 0, "Quen": 0, "Axii": 0}
last_action_time = 0
active_gesture = None

# --- Helper Functions ---

def get_landmark_coords(hand_landmarks, landmark_index):
    """Gets the x, y coordinates of a specific landmark."""
    lm = hand_landmarks.landmark[landmark_index]
    return lm.x, lm.y

def is_finger_extended(hand_landmarks, finger_tip_index, finger_pip_index):
    """
    Checks if a finger is extended by comparing the y-coordinate of the tip
    with the y-coordinate of the PIP joint. Assumes an upright hand.
    """
    tip_x, tip_y = get_landmark_coords(hand_landmarks, finger_tip_index)
    pip_x, pip_y = get_landmark_coords(hand_landmarks, finger_pip_index)
    return tip_y < pip_y

def is_thumb_tucked(hand_landmarks):
    """Checks if the thumb is tucked in, close to the palm."""
    thumb_tip_x, thumb_tip_y = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP)
    index_mcp_x, index_mcp_y = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_MCP)
    distance = math.hypot(thumb_tip_x - index_mcp_x, thumb_tip_y - index_mcp_y)
    return distance < 0.1

def is_thumb_extended(hand_landmarks):
    """Checks if the thumb is extended outwards from the fist."""
    thumb_tip_x, _ = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP)
    thumb_ip_x, _ = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.THUMB_IP)
    # This logic assumes a right hand in a flipped (selfie) view.
    return thumb_tip_x < thumb_ip_x

def is_aard_gesture(hand_landmarks):
    """Detects the Aard sign: Index and Middle extended, others curled."""
    try:
        index_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
        middle_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        thumb_is_tucked = is_thumb_tucked(hand_landmarks)
        return index_extended and middle_extended and ring_curled and pinky_curled and thumb_is_tucked
    except Exception:
        return False

def is_igni_gesture(hand_landmarks):
    """Detects the Igni sign: Fist with thumb extended."""
    try:
        index_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
        middle_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        thumb_is_extended = is_thumb_extended(hand_landmarks)
        return index_curled and middle_curled and ring_curled and pinky_curled and thumb_is_extended
    except Exception:
        return False

def is_quen_gesture(hand_landmarks):
    """Detects the Quen sign: Thumb and Index tips touching, other three fingers extended."""
    try:
        middle_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        
        # Check distance between thumb tip and index finger tip
        thumb_tip_x, thumb_tip_y = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP)
        index_tip_x, index_tip_y = get_landmark_coords(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        distance = math.hypot(thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y)
        
        # A small distance indicates the tips are touching. This value is more lenient.
        tips_touching = distance < 0.07

        # The explicit check for a curled index finger has been removed as it was too strict.
        # If the tips are touching, the index finger is inherently not fully extended.
        return middle_extended and ring_extended and pinky_extended and tips_touching
    except Exception:
        return False

def is_axii_gesture(hand_landmarks):
    """Detects the Axii sign: Thumb and Index extended ('L' shape), others curled."""
    try:
        thumb_is_extended = is_thumb_extended(hand_landmarks)
        index_extended = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
        middle_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky_curled = not is_finger_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        return thumb_is_extended and index_extended and middle_curled and ring_curled and pinky_curled
    except Exception:
        return False

def cast_sign(key, sign_name):
    """Simulates pressing a key to cast a sign."""
    print(f"--- {sign_name.upper()}! Casting sign with key '{key}' ---")
    keyboard.press(key)
    time.sleep(0.1)
    keyboard.release(key)
    print("--------------------------------------")

# --- Main Loop ---
print("Controller active. Show a gesture to the camera.")
print("Press 'q' in the preview window to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    current_gesture_in_frame = None
    status_text = "No Gesture"
    status_color = (0, 0, 255) # Red

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for gestures in order of priority to prevent conflicts
            if is_quen_gesture(hand_landmarks):
                current_gesture_in_frame = "Quen"
                status_text = "Quen Gesture Detected"
                status_color = (0, 255, 255) # Yellow
            elif is_aard_gesture(hand_landmarks):
                current_gesture_in_frame = "Aard"
                status_text = "Aard Gesture Detected"
                status_color = (0, 255, 0) # Green
            elif is_igni_gesture(hand_landmarks):
                current_gesture_in_frame = "Igni"
                status_text = "Igni Gesture Detected"
                status_color = (0, 165, 255) # Orange
            elif is_axii_gesture(hand_landmarks):
                current_gesture_in_frame = "Axii"
                status_text = "Axii Gesture Detected"
                status_color = (255, 0, 255) # Magenta
    
    # Updated logic for counting gestures
    if current_gesture_in_frame:
        if current_gesture_in_frame == active_gesture:
            gesture_counters[active_gesture] += 1
        else:
            active_gesture = current_gesture_in_frame
            for sign in gesture_counters:
                gesture_counters[sign] = 0
            gesture_counters[active_gesture] = 1
    else:
        active_gesture = None
        for sign in gesture_counters:
            gesture_counters[sign] = 0

    # Check if a sign should be cast
    if active_gesture and (time.time() - last_action_time) > ACTION_COOLDOWN_SECONDS:
        if gesture_counters[active_gesture] >= GESTURE_COOLDOWN_FRAMES:
            key_to_press = globals()[f"{active_gesture.upper()}_KEY"]
            cast_sign(key_to_press, active_gesture)
            last_action_time = time.time()
            for sign in gesture_counters:
                gesture_counters[sign] = 0
            active_gesture = None

    # Display status on the screen
    cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
    cv2.putText(image, "Press 'q' to quit", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if (time.time() - last_action_time) <= ACTION_COOLDOWN_SECONDS:
        cv2.putText(image, "COOLDOWN", (CAM_WIDTH - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Witcher 3 Gesture Controller', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Shutting down controller.")
cap.release()
cv2.destroyAllWindows()
hands.close()
