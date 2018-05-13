class HOGSVNClassifier:
	def __init__(self, svn, hog):
		self.svn = svn
		self.hog = hog

	def classify(self, image):
		