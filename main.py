import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# You can use this in a video file by removing 0 (default camera) and replacing it with the path of your video
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
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

    for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('Human Detector', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
