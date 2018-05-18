from preprocessor import Preprocessor

class RemoveRed(Preprocessor):
	def remove_red(self, img):
		img[:,:,2] = 0
		return img

	def process(self, image):
		image = image.copy()
		return self.remove_red(image)
