import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
 
cap = cv2.VideoCapture(0)

while True:
    true, im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in face:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        gray_temp = gray[y:y+h, x:x+w]
                
        smile = smileCascade.detectMultiScale(gray_temp, scaleFactor= 1.3, minNeighbors=5, minSize=(30, 30))
        
        for i in smile:
            if len(smile)>1:
                cv2.putText(im,"Smiling",(x,y-50),cv2.FONT_HERSHEY_PLAIN,
                    2,(0,0,204),3,cv2.LINE_AA)
               
    cv2.imshow('RESULT', im)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()