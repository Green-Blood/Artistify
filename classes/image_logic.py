# import the necessary packages
import cv2
from classes.enhancement import *
from classes.parts_detection import *


class ImageLogic:
    img = None
    fImg = None

    def __init__(self):
        self.img = None

    # Apply image processing techniques
    def apply_techniques(self, id, number):
        # Read image
        img = cv2.imread(f"./static/images/{self.img}")
        # Apply function by id of the clicked button
        func = switch(id, number)
        # Get filtered image
        filtered_img = func(img)
        # Save filtered image
        self.fImg = f"./static/images/f{self.img}"
        cv2.imwrite(self.fImg, filtered_img)

    # Apply parts detection function
    def apply_parts(self, part_type):
        # Read image
        img = cv2.imread(f"./static/images/{self.img}")
        # Apply parts detection function
        parts_detector(self.img, part_type)
        # Save image
        self.fImg = f"./static/images/f{self.img}"

    # Set filtered image
    def set_filtered(self):
        # Read image
        img = cv2.imread(f"./static/images/{self.img}")
        # Apply image processing technique
        filtered_img = do_gaussian(img)
        # Save image
        self.fImg = f"./static/images/f{self.img}"
        cv2.imwrite(self.fImg, filtered_img)

    # Get filtered image
    def get_filtered(self):
        return self.fImg

    # Set uploaded image
    def set_image(self, image):
        self.img = image

    # Get uploaded image
    def get_image(self):
        return self.img
