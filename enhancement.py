# import the necessary packages
import cv2
import numpy as np


# Do Grayscale filtering
def do_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Do gaussian blur filtering using 5x5 kernel
def do_gaussian(img):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)


# Do binary filtering
def do_binary(img):
    gray = do_gray(img)
    return cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]


# Do canny edge detection with 50 and 150 threshold values
def do_canny(img):
    # Do Grayscale filtering
    gray = do_gray(img)
    # Do gaussian blur filtering using 5x5 kernel
    gauss = do_gaussian(gray)
    # Apply canny edge detection
    return cv2.Canny(gauss, 50, 150)


# Do image sharpening image processing technique
def do_sharpening(img):
    # Create our sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # applying the sharpening kernel to the input image.
    return cv2.filter2D(img, -1, kernel)


# Do negative image processing technique
def do_negative(img):
    # Do grayscale filtering
    gray = do_gray(img)
    # Return the negative of gray
    return 255 - gray


# Create sepia filtered image
def do_sepia(img):
    # Create sepia kernel
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    # applying the sepia kernel to the input image.
    return cv2.filter2D(img, -1, kernel)


# Create emboss filtered image
def do_emboss(img):
    # Create emboss kernel
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    # applying the emboss kernel to the input image.
    return cv2.filter2D(img, -1, kernel)


# Stylization function to create the effect of smoothing out the colors and the image.
def do_cartoon(img):
    return cv2.stylization(img, sigma_s=150, sigma_r=0.25)


# Create sketch like filtered image
def do_sketch(img):
    sketch1, color_sketch2 = cv2.pencilSketch(img, sigma_s=100, sigma_r=0.5, shade_factor=0.02)
    return sketch1


# Create colored sketch like filtered image
def do_color_sketch(img):
    sketch1, color_sketch2 = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.5, shade_factor=0.02)
    return color_sketch2


# Switch function to choose which of the technique to apply to the image, based on the button clicked
def switch(id, num):
    processing_switcher = {
        0: do_gray,
        1: do_binary,
        2: do_gaussian,
        3: do_sharpening,
        4: do_sepia,
        5: do_emboss,
        6: do_canny,
        7: do_negative,
        8: do_cartoon,
        9: do_sketch,
        10: do_color_sketch,

    }
    switcher = None
    if num == 0:
        switcher = processing_switcher.get(id, lambda: 'Invalid')
    return switcher
