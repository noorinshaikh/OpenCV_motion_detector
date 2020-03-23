import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image=cv2.imread("us.jpg")
grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(grey_image,scaleFactor=1.05,minNeighbors=5)
for x,y,w,h in faces:
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
    #cropped_img = image[y:y+h,x:x+w]
resized_image=cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))

cv2.imshow("detector",resized_image)
cv2.waitKey(3000)
cv2.destroyAllWindows()
