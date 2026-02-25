import cv2

cap = cv2.VideoCapture(0)  # if not working, try 1

if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try index 1 or close apps using the camera.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Webcam - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()