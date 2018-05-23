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
		for x, y, w, h in regions:
		    #get slice at box:
		    window = image[y:y+h, x:x+w, :3]
		    #refactor to abstract out hog
		    window = cv2.resize(window, (80,80))      		
			#     plt.imshow(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
			#     plt.show()
			#     print prob[1]
		    prob = self.classifier.classify_with_probability(window)
		    if prob >= 0:  
		        detections.append(((x,y,w,h), prob))

		return detections

	def display_signs(self, detections, image):
		image = image.copy()
		print("{} detections".format(len(detections)))
		for box, confidence in detections:
			self.draw_boundary_on_image(box, confidence, image)
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.show()


	def draw_boundary_on_image(self, boundary, confidence, image):	
		x, y, w, h = boundary	
		cv2.rectangle(image, (x,y),(x+w, y+h), (0, 255, 0), 2)
		print(confidence)
		cv2.putText(image,"{0:.4f}".format(confidence), (x+5,y+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 200))

