class RemoveRed(Preprocessor):
	def remove_red(img):
	    img[:,:,2] = 0
	    return img