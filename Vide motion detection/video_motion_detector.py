import cv2,time

first_frame=None
i=0
video=cv2.VideoCapture(0)
while True:
    check, frame=video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None or i<5:
        first_frame=gray_frame
        i+=1
        continue
    delta_frame=cv2.absdiff(first_frame,gray_frame)
    threshold_delta=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_delta=cv2.dilate(threshold_delta,None,iterations=2)
    (cnts,_)=cv2.findContours(threshold_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow("Capturing", threshold_delta)
    cv2.imshow("Capturing_delta",delta_frame)
    cv2.imshow("Capturing_frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()