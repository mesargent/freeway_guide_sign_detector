class Detector:
	def __init__(self, params, preprocessors, region_finder, classifier):
		self.preprocessors = preprocessors
		self.region_finder = region_finder
		self.classifier = classifier
		self.params = params

	def detect_photo(self, image):
		image = self.preprocess(image)
		regions = self.region_finder.find_regions(image)
		detections = []
		for region in regions:
			if self.classifier.classify(region):
				detections.add(region)
		return region	

	def detect_video(self, video):
		pass

	def preprocess(self, image, preprocessors):
		for preprocessor in self.preprocessors:
			image = preprocessor.process(image)
		return image

	