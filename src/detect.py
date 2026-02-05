# src/detect.py
"""
Face detection and locking with action detection.
Uses the lock.py module for face locking functionality.

Run:
python -m src.detect

Keys:
q : quit
r : reload DB
l : release lock
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .haar_5pt import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .recognize import (
    HaarFaceMesh5pt,
    FaceDet,
    FaceDBMatcher,
    load_db_npz,
)
from .lock import (
    LockState,
    detect_actions,
    _face_center_x,
    _eye_nose_dist,
    _mouth_width_ratio,
    LOCK_SIM_THRESHOLD,
    TRACK_SIM_THRESHOLD,
    LOCK_RELEASE_SEC,
)

# Config
DB_PATH = Path("data/db/face_db.npz")
HISTORY_DIR = Path("data/lock_history")


def main():
    db_path = DB_PATH
    db = load_db_npz(db_path)
    if not db:
        print("No enrolled identities. Run: python -m src.enroll")
        return

    names = sorted(db.keys())
    print("Enrolled identities:", ", ".join(names))
    target = input("Enter identity to lock:").strip()
    if not target or target not in db:
        print(f"Unknown identity '{target}'. Choose from: {', '.join(names)}")
        return

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(input_size=(112, 112), debug=False)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)

    lock = LockState()
    locked_embedding = db[target].reshape(-1).astype(np.float32)

    cap = cv2.videoCapture(1)Capture(2)pture(1)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print("Face Locking. q=quit, r=reload DB, l=release lock")
    print(f"Waiting for '{target}' to appear...")

    t0 = time.time()
    fps_t0 = t0
    fps_n = 0
    fps = 0.0

    # Track last detected actions for display
    last_actions = []
    action_display_time = 2.0  # seconds to display action

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()
        h, w = vis.shape[:2]

        best_face: Optional[Tuple[FaceDet, float, object, np.ndarray]] = None

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned).embedding
            mr = matcher.match(emb)
            sim_to_target = float(np.dot(emb, locked_embedding))

            if not lock.is_locked():
                if mr.name == target and mr.accepted and sim_to_target >= LOCK_SIM_THRESHOLD:
                    if best_face is None or sim_to_target > best_face[1]:
                        best_face = (f, sim_to_target, mr, emb)
            else:
                if sim_to_target >= TRACK_SIM_THRESHOLD:
                    if best_face is None or sim_to_target > best_face[1]:
                        best_face = (f, sim_to_target, mr, emb)

        # Track actions before detection for display
        actions_before = len(lock.last_action_time) if lock.is_locked() else 0

        if lock.is_locked():
            if best_face is not None:
                f, sim, mr, _ = best_face
                lock.last_seen_time = now
                
                # Detect actions
                detect_actions(lock, f, now)
                
                # Check if new action was detected
                actions_after = len(lock.last_action_time)
                if actions_after > actions_before:
                    # Get the most recent action
                    recent_action = max(lock.last_action_time.items(), key=lambda x: x[1])
                    last_actions.append((recent_action[0], now))
                
                # Draw locked face - GREEN color (0, 255, 0)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 3)
                cv2.putText(vis, f"LOCKED: {lock.identity}", (f.x1, max(0, f.y1 - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(vis, f"sim={sim:.2f}", (f.x1, max(0, f.y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw landmarks - GREEN color
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Display recent actions below the face box - GREEN color
                last_actions = [(action, t) for action, t in last_actions if now - t < action_display_time]
                for idx, (action, t) in enumerate(last_actions[-3:]):  # Show last 3 actions
                    action_text = action.replace("_", " ").upper()
                    cv2.putText(vis, f"ACTION: {action_text}", 
                               (f.x1, f.y2 + 25 + idx * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw all OTHER faces (not the locked one) with proper labels
                for other_f in faces:
                    # Skip the locked face
                    if (other_f.x1 == f.x1 and other_f.y1 == f.y1 and 
                        other_f.x2 == f.x2 and other_f.y2 == f.y2):
                        continue
                    
                    # Recognize this face
                    aligned, _ = align_face_5pt(frame, other_f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned).embedding
                    mr = matcher.match(emb)
                    label = mr.name if mr.name else "Unknown"
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)  # Green if known, Red if unknown
                    cv2.rectangle(vis, (other_f.x1, other_f.y1), (other_f.x2, other_f.y2), color, 2)
                    cv2.putText(vis, label, (other_f.x1, max(0, other_f.y1 - 8)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                if now - lock.last_seen_time > LOCK_RELEASE_SEC:
                    lock.release()
                    print("Lock released (face not seen for too long).")
                    last_actions.clear()
                else:
                    cv2.putText(vis, f"LOCKED: {lock.identity} (searching...)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    
                    # Even when searching, show all detected faces
                    for other_f in faces:
                        aligned, _ = align_face_5pt(frame, other_f.kps, out_size=(112, 112))
                        emb = embedder.embed(aligned).embedding
                        mr = matcher.match(emb)
                        label = mr.name if mr.name else "Unknown"
                        color = (0, 255, 0) if mr.accepted else (0, 0, 255)  # Green if known, Red if unknown
                        cv2.rectangle(vis, (other_f.x1, other_f.y1), (other_f.x2, other_f.y2), color, 2)
                        cv2.putText(vis, label, (other_f.x1, max(0, other_f.y1 - 8)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            if best_face is not None:
                f, sim, mr, emb = best_face
                lock.identity = target
                lock.embedding = emb.copy()
                lock.lock_time = now
                lock.last_seen_time = now
                lock.prev_center_x = _face_center_x(f.kps)
                lock.prev_eye_nose_dist = _eye_nose_dist(f.kps)
                lock.prev_mouth_ratio = _mouth_width_ratio(f.kps)
                ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
                lock.history_path = HISTORY_DIR / f"{target.lower()}_history_{ts}.txt"
                lock.history_file = open(lock.history_path, "w", encoding="utf-8")
                lock.history_file.write("timestamp\taction_type\tdescription\n")
                lock.write_action("lock_acquired", f"locked onto {target}")
                print(f"Locked onto {target}. History: {lock.history_path}")
                last_actions.clear()
            else:
                for f in faces:
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned).embedding
                    mr = matcher.match(emb)
                    label = mr.name if mr.name else "Unknown"
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, label, (f.x1, max(0, f.y1 - 8)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now
        
        cv2.putText(vis, f"fps: {fps:.1f} | target: {target}", (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if lock.is_locked():
            cv2.putText(vis, "LOCKED - actions recorded", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Face Locking System", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            db = load_db_npz(db_path)
            matcher.reload(db_path)
            print(f"DB reloaded: {len(db)} identities")
        elif key == ord("l"):
            if lock.is_locked():
                lock.release()
                print("Lock released (manual).")
                last_actions.clear()

    lock.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
