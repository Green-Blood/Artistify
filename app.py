# import the necessary packages
from flask import Flask, render_template, redirect, request, jsonify, url_for
from flask_cors import CORS
from image_logic import ImageLogic
from enhancement import *
from face_detection import Face
from age import Age
from parts_detection import *
import numpy as np
import logging
import base64
import os
import cv2

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)
img = ImageLogic()


# Route for index page
@app.route("/")
def home():
    return render_template('index.html')


# Route for about page
@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/<name>")
def name_route(name):
    return f"<h1>{name}</h1>"


# Image upload
@app.route("/image", methods=["POST"])
def image_route():
    # Check if POST request has the file
    file = request.files['file']
    img.set_image(file.filename)
    file.save(os.path.join("static", "images", img.get_image()))
    return render_template('index.html', image=img.get_image())

# Route for age detector page
@app.route("/age", methods=["POST", "GET"])
def age_route():
    # Check if the image have been uploaded
    if img.get_image() is None:
        return redirect(url_for('home', error="Error! Choose file first"))
    Age.age_detector(img.get_image())
    result = f"static/images/a{img.get_image()}"
    return render_template('age.html', image=img.get_image(), result=result)


# Route for parts detector page
@app.route("/parts", methods=["POST", "GET"])
def parts_route():
    if request.method == "POST":
        json = request.json
        img.apply_parts(int(json['mt']))
        return jsonify(img.get_filtered())

    # Check if the image have been uploaded
    if img.get_image() is None:
        return redirect(url_for('home', error="Error! Choose file first"))

    items = [
        {
            "id": 0,
            "title": "Mouth",
        },
        {
            "id": 1,
            "title": "Right eyebrow"
        },
        {
            "id": 2,
            "title": "Left eyebrow"
        },
        {
            "id": 3,
            "title": "Right eye"
        },
        {
            "id": 4,
            "title": "Left eye"
        },
        {
            "id": 5,
            "title": "Nose"
        },
        {
            "id": 6,
            "title": "Jawline"
        },
    ]

    return render_template('parts.html', items=items, image=img.get_image())


# Route for gallery page
@app.route("/gallery")
def gallery_route():
    images = []
    for filename in os.listdir('static/images'):
        if filename.endswith(".jpg") and filename != "noimage.png":
            images.append(os.path.join('static/images', filename))
        else:
            continue
    return render_template("gallery.html", image=images)


# Route for image processing techniques page
@app.route("/processing", methods=["POST", "GET"])
def process_route():
    if request.method == "POST":
        json = request.json
        img.apply_techniques(int(json['mt']), 0)
        return jsonify(img.get_filtered())

    if img.get_image() is None:
        return redirect(url_for('home', error="Error! Choose file first"))
    items = [
        {
            "id": 0,
            "title": "Gray scale"
        },
        {
            "id": 1,
            "title": "Binary"
        },
        {
            "id": 2,
            "title": "Blur"
        },
        {
            "id": 3,
            "title": "Sharpening"
        },
        {
            "id": 4,
            "title": "Sepia"
        },
        {
            "id": 5,
            "title": "Emboss"
        },
        {
            "id": 6,
            "title": "Canny"
        },

        {
            "id": 7,
            "title": "Negative"
        },
        {
            "id": 8,
            "title": "Cartoon"
        },
        {
            "id": 9,
            "title": "Sketch"
        },
        {
            "id": 10,
            "title": "Color Sketch"
        },

    ]
    return render_template('processing.html', image=img.get_image(), items=items)


# Route for face detector page
@app.route('/face')
def face_detection():
    return render_template("face_detection.html", image=img.get_image())


# Route to upload photo to detect the face
@app.route('/upload', methods=['POST'])
def upload_file():
    # Read file
    file = request.files['image']
    # Save file
    filename = 'static/images' + file.filename
    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Detect faces
    faces = Face.detect_faces(image)

    if len(faces) == 0:
        face_detected = False
        num_faces = 0
        to_send = ''
    else:
        face_detected = True
        num_faces = len(faces)

        # Draw a rectangle
        for item in faces:
            Face.draw_rectangle(image, item['rect'])

        # Save
        cv2.imwrite(filename, image)

        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('face_detection.html', faceDetected=face_detected,
                           num_faces=num_faces, image_to_show=to_send,
                           init=True)


if __name__ == '__main__':
    app.run(debug=True)
