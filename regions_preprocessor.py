from preprocessor import Preprocessor

import matplotlib.pyplot as plt
import numpy as np
import cv2

class RegionsPreprocessor(Preprocessor):

	def process(self, image):
		lower = np.array(self.config["preprocessing_params"]["colors_lower"], dtype="uint8")
		upper = np.array(self.config["preprocessing_params"]["colors_upper"], dtype="uint8")
		mask = cv2.inRange(image, lower, upper)
		pimage = cv2.bitwise_and(image, image, mask = mask)
		imgray = cv2.cvtColor(pimage,cv2.COLOR_BGR2GRAY)
		flag, binary_image = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return binary_image
