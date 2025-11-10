import cv2
import mediapipe as mp

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # If hands are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks on hand
                mp_draw.draw_landmarks(
                    frame, 
                    handLms, 
                    mp_hands.HAND_CONNECTIONS
                )

        # Show output
        cv2.imshow("Hand Tracking", frame)

        # Close window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
