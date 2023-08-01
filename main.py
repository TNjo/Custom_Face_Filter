import cv2
import cvzone

cap = cv2.VideoCapture(0) # 0 for webcam
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # face detection model
overlay = cv2.imread('t4.1.png', cv2.IMREAD_UNCHANGED) # overlay image

while True:
    _, frame = cap.read() # read frame
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale
    faces = cascade.detectMultiScale(gray_scale) # detect faces
    for x, y, w, h in faces: # draw rectangle around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw rectangle
        overlay_resize = cv2.resize(overlay, (int(w*1.0), int(h*1.0))) # resize overlay image
        frame = cvzone.overlayPNG(frame, overlay_resize, [x, y-5]) # overlay image

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
