import numpy as np
from .face_tracker import FaceTracker
from .history_logger import start_history, log_action
from .actions import detect_horizontal_movement, detect_smile

tracker = FaceTracker(target_name="Gloria")
history_file = None
prev_center = None


def handle_face(name, bbox, landmarks=None):
    global history_file, prev_center
    
    detected_action = None
    status = tracker.update(name, bbox)

    if status == "LOCKED" and history_file is None:
        history_file = start_history(name)
        log_action(history_file, "LOCKED", "Face successfully locked")

    if tracker.locked:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        current_center = (cx, cy)

        # Smile detection
        if landmarks is not None and history_file:
            # kps indices: 0=LE, 1=RE, 2= Nose, 3=MouthLeft, 4=MouthRight
            # mouth_width = dist(3, 4)
            # face_width = dist(0, 1) (outer eyes)
            mw = np.linalg.norm(landmarks[3] - landmarks[4])
            fw = np.linalg.norm(landmarks[0] - landmarks[1])
            
            # Use 'effective' face width for ratio calculation if needed, 
            # but detect_smile uses simple ratio.
            if detect_smile(mw, fw):
                ratio = mw / fw
                log_action(history_file, "smile", f"Smile detected (ratio={ratio:.3f})")
                detected_action = "SMILE"

        if prev_center:
            movement = detect_horizontal_movement(prev_center, current_center)
            if movement:
                log_action(history_file, movement)
                detected_action = movement

        prev_center = current_center

    return detected_action
