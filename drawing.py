import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
canvas = None
prev_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 800:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if prev_point:
                cv2.line(canvas, prev_point, (cx, cy),
                         (255, 255, 255), 4)

            prev_point = (cx, cy)
        else:
            prev_point = None
    else:
        prev_point = None

    output = cv2.add(frame, canvas)

    cv2.imshow("Smooth Air Drawing", output)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
