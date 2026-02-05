import cv2
import os
import sys
import contextlib


def _get_camera_index(default: int = 1) -> int:
    s = os.environ.get("CAMERA_INDEX")
    if s is None:
        return default
    try:
        return int(s)
    except Exception:
        return default


@contextlib.contextmanager
def _suppress_stderr():
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)
        os.close(devnull)


def main():
    cam_idx = _get_camera_index(1)
    with _suppress_stderr():
        cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened (index {cam_idx}). Try changing CAMERA_INDEX environment variable.")

    print("Camera test. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Camera Test", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
