import cv2
import numpy

class RegionFinder:
	def __init__(self, config, preprocessors):
		self.config = config
		self.preprocessors = preprocessors

	def find_regions(self, image):
		for processor in preprocessors:
			image = processor.process(image)

		contours, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		return contours