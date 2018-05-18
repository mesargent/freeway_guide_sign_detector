import cv2
import numpy

class RegionFinder:
	def __init__(self, config, preprocessors):
		self.config = config
		self.preprocessors = preprocessors

	def find_regions(self, image):
		for processor in self.preprocessors:
			image = processor.process(image)
		_, contours, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		regions = [cv2.boundingRect(c) for c in contours]
		regions = [b for b in regions if b[2]*b[3] > 600]
		print(len(regions))
		return regions