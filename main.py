# open-cv-python
import cv2

# Getting haar cascade face and eyes data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# You can use this in a video file by removing 0 (default camera) and replacing it with the path of your video
video = cv2.VideoCapture(0)

while True:
    # Open camera
    check, frame = video.read()
    
    # haar cascade configurations
    eyes = eye_cascade.detectMultiScale(
        frame, 
        scaleFactor = 1.2,
        minNeighbors = 5
    )

    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor = 1.2, 
        minNeighbors = 5
    )

    # Creating a following rectangle on the eyes and faces
    for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display detector on a window
    cv2.imshow('Human Detector', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
