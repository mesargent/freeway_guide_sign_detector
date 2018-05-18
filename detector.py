import cv2
import matplotlib.pyplot as plt
import numpy as np

class Detector:
	def __init__(self, params, preprocessors, region_finder, classifier):
		self.preprocessors = preprocessors
		self.region_finder = region_finder
		self.classifier = classifier
		self.params = params

	def detect(self, image):
		regions = self.region_finder.find_regions(image)
		detections = []
		for region in regions:
			if self.classifier.classify(region):
				detections.append(region)
		return detections

	def display_signs(self, detections, image):
		image = image.copy()
		print("{} detections".format(len(detections)))
		for detection in detections:
			self.draw_boundary_on_image(detection, image)
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.show()


	def draw_boundary_on_image(self, boundary, image):	
		x, y, w, h = boundary	
		cv2.rectangle(image, (x,y),(x+w, y+h), (0, 255, 0), 2)

