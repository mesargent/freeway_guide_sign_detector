import os.path

from sklearn import svm
from sklearn.externals import joblib

import numpy as np
import cv2

class HOGSVMClassifier:
	def __init__(self, config):
		minDim = 80
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		dims = (minDim, minDim)
		#TODO get params from config file
		self.hog = cv2.HOGDescriptor(dims, blockSize, blockStride, cellSize, nbins)
		self.svm = get_trained_SVM()

	def classify(self, image):
		# features = self.hog.compute(image.astype(np.uint8))
		# return svm.predict(image)
		return True

	def get_trained_SVM(self):
		if os.path.exists(config[svm_params][model_path]):
			return joblib.load(config[svm_params][model_path])
		return self.train_new_SVM()

	def train_new_SVM(self):
		svm = svm.SVC(kernel='linear', C = 1.0, probability=True)
		#train svm

		joblib.dump(config[svm_params][model_path])
		self.print_svm_meta_data()

		return svm

	def print_svm_meta_data(self):
		pass
