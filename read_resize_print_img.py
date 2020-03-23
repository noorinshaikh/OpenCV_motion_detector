import cv2

image=cv2.imread("galaxy-image.jpg",0)
print(type(image))
print(image.shape)
print(image.ndim)
new_image=cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))

cv2.imshow("Galaxy",new_image)
cv2.imwrite("resized_img.jpg",new_image)
cv2.waitKey(3000)
cv2.destroyAllWindows()