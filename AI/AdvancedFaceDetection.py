import cv2
import imutils as i
alg="haarcascade_frontalface_default.xml"
haar_cascade=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(1)
while True:
    _,img=cam.read()
    gImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(gImg,1.3,4)
    for (x,y,w,h)in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    img=i.resize(img,width=1000)
    cv2.imshow("Face Detection",img)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
