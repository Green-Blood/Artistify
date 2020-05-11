# import the necessary packages
from imutils import face_utils
import imutils
import dlib
import cv2


def parts_detector(img, part_type):
    # Trained models path
    shape_predictor = "./trainings/shape_predictor_68_face_landmarks.dat"
    # Image path
    img_path = f"./static/images/{img}"
    # Parts of the face
    names = ["mouth", "right_eyebrow", "left_eyebrow", "right_eye", "left_eye", "nose", "jaw"]
    # Take the needed part
    name = names[part_type]

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        face = face_utils.FACIAL_LANDMARKS_IDXS[name]
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (192, 0, 192), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[face[0]:face[1]]:
            cv2.circle(clone, (x, y), 4, (255, 0, 0), -1)
        cv2.imwrite(f"./static/images/f{img}", clone)




