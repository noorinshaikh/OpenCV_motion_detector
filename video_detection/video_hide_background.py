import cv2,time
import numpy as np

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
while True:
    check, frame=video.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(blurred_img, scaleFactor=1.10, minNeighbors=2)
    for x, y, w, h in face:

        sub_frame = gray_frame[y:y + h, x:x + w]
        blurred_img = cv2.GaussianBlur(blurred_img, (23,23),0)
        blurred_img[y:y+sub_frame.shape[0], x:x+sub_frame.shape[1]]=sub_frame
        blurred_img = cv2.rectangle(blurred_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow("Capturing_hidebg",blurred_img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
