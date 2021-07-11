import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)


while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    Text ="Person Not detected"
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        Text = "person detected"
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        img = cv2.putText(img,Text,(120,100),font,0.5,(0,255,255))
    print(Text)
    cv2.imshow("facedetection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
   
cam.release()
cv2.destroyAllWindows()
