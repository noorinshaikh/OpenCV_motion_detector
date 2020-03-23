import cv2
import glob

images_list=glob.glob("multi_image/*.jpg")
count=0
for image in images_list:
    img=cv2.imread(image,0)
    img_resize=cv2.resize(img,(100,100))
    cv2.imshow("image"+str(count),img_resize)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    path=image.split('/')
    cv2.imwrite(path[0]+"/resized_"+path[1],img_resize)
    count+=1
