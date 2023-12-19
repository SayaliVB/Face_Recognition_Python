import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img=cv2.imread("photo.jpg") #no 0 or 1 or -1 means the orignal color
img=cv2.imread("SAM1.JPG")
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayScale

#instead of these two commands, we can directly store grayScaleimage but i want to display colored image

faces= face_cascade.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=5)
print(faces) #numpy ndarray

count=0

for x,y,w,h in faces:
    count=count+1
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,45,255),5)
    roi_gray = gray_img[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.imwrite(str(count)+".jpg", roi_gray)
    '''
    cv2.imshow("Image",roi_color)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    '''


#new_img= cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

new_img= cv2.resize(img, (1000,500))

cv2.imshow("Image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
