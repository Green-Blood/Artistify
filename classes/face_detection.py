# import the necessary packages
import cv2


class Face:
    def detect_faces(img):
        # List of the faces
        faces_list = []
        # Convert the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Load face detector
        face_cascade = cv2.CascadeClassifier('trainings/opencv-files/lbpcascade_frontalface.xml')
        # Detect multiscale images
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

        # If no face detected, return empty list
        if len(faces) == 0:
            return faces_list

        for i in range(0, len(faces)):
            (x, y, w, h) = faces[i]
            face_dict = {'face': gray[y:y + w, x:x + h], 'rect': faces[i]}
            faces_list.append(face_dict)
        # Return the face image area and the face rectangle
        return faces_list

    # Draw rectangle on the face
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)