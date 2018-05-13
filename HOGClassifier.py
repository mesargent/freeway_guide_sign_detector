class HOGSVNClassifier:
	def __init__(self, svn, hog):
		self.svn = svn
		self.hog = hog

	def classify(self, image):
		features = hog.compute(img[:,:,:3])
		return svn.predict(image)