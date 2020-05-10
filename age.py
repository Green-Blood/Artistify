# import the necessary packages
import cv2


# Class for age detection
class Age:
    # Function to find and highlight the face
    def highlight_face(net, frame, conf_threshold=0.7):
        f_opecv_dnn = frame.copy()
        f_height = f_opecv_dnn.shape[0]
        f_width = f_opecv_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(f_opecv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        f_box = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * f_width)
                y1 = int(detections[0, 0, i, 4] * f_height)
                x2 = int(detections[0, 0, i, 5] * f_width)
                y2 = int(detections[0, 0, i, 6] * f_height)
                f_box.append([x1, y1, x2, y2])
                cv2.rectangle(f_opecv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(f_height / 150)), 8)
        return f_opecv_dnn, f_box

    # Detects age and gender
    def age_detector(img):
        face_photo = "./trainings/age/opencv_face_detector.pbtxt"
        face_model = "./trainings/age/opencv_face_detector_uint8.pb"
        age_proto = "./trainings/age/age_deploy.prototxt"
        age_model = "./trainings/age/age_net.caffemodel"
        gender_proto = "./trainings/age/gender_deploy.prototxt"
        gender_model = "./trainings/age/gender_net.caffemodel"
        img_path = f"./static/images/{img}"
        model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        gender_list = ['Male', 'Female']

        face_net = cv2.dnn.readNet(face_model, face_photo)
        age_net = cv2.dnn.readNet(age_model, age_proto)
        gender_net = cv2.dnn.readNet(gender_model, gender_proto)

        video = cv2.VideoCapture(img_path if img_path else 0)
        padding = 20

        while cv2.waitKey(1) < 0:
            ret, frame = video.read()
            if not ret:
                cv2.waitKey()
                break
            result_img, face_boxes = Age.highlight_face(face_net, frame)
            if not face_boxes:
                print("No face detected")
            for faceBox in face_boxes:
                face = frame[max(0, faceBox[1] - padding):
                             min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                            :min(faceBox[2] + padding,
                                                                                 frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print(f'Gender: {gender}')

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(result_img, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imwrite(f"./static/images/a{img}", result_img)
