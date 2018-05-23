import os.path
from random import shuffle

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2

import data_generator as dg

class HOGSVMClassifier:
	def __init__(self, config):
		self.config = config
		self.minDim = 80
		self.blockSize = (16,16)
		self.blockStride = (8,8)
		self.cellSize = (8,8)
		self.nbins = 9
		self.dims = (self.minDim, self.minDim)
		#TODO get params from config
		self.hog = cv2.HOGDescriptor(self.dims, self.blockSize, self.blockStride, self.cellSize, self.nbins)
		self.svm = self.get_trained_SVM()

	def classify(self, image):
		features = self.hog.compute(image.astype(np.uint8))
		return self.svm.predict(features)

	def classify_with_probability(self, image):
		features = self.hog.compute(image.astype(np.uint8))
		return self.svm.predict_proba(features.reshape(1, -1))[0][1]

	def get_trained_SVM(self):
		if os.path.exists(self.config["svm_params"]["model_path"]):
			return joblib.load(self.config["svm_params"]["model_path"])
		lsvm = self.train_new_SVM()
		self.print_svm_meta_data(lsvm)
		return 

	def train_new_SVM(self):
		#TODO refactor to use config
		#TODO refactor to abstract training of classifiers into a separate class
		pimages = dg.getImagesFromJSON(open("labels.json").read())
		print(len(pimages), "positive images")

		#get negative images, use ratios found for positive images to match
		nimages = dg.getRandomMultiple("sun_images", 200, 0)
		partialsignimgs = dg.getRandomMultiple("partial", 200, 0)
		nimages = nimages+partialsignimgs
		nimages = [x[1] for x in nimages]

		print(len(nimages), "negative images")

		pdata = self.getFeaturesWithLabel(pimages, self.hog, self.dims, 1)
		ndata = self.getFeaturesWithLabel(nimages, self.hog, self.dims, 0)

		data = pdata + ndata
		shuffle(data)

		feat, labels = map(list, zip(*data))
		feat = [x.flatten() for x in feat]

		sample_size = len(feat)
		train_size = int(round(0.8*sample_size))

		train_feat = np.array(feat[:train_size], np.float32)
		test_feat = np.array(feat[train_size: sample_size], np.float32)
		train_label = np.array(labels[:train_size])
		test_label = np.array(labels[train_size:sample_size])

		lsvm = svm.SVC(kernel='linear', C = 1.0, probability=True)
		lsvm.fit(train_feat, train_label)

		joblib.dump(lsvm, self.config["svm_params"]["model_path"])


		return lsvm

	def getFeaturesWithLabel(self, imageData, hog, dims, label):
		data = []
		for img in imageData: 
		    img = cv2.resize(img, dims)

		    #for images with transparency layer, reduce to 3 layers
		    feat = hog.compute(img[:,:,:3])
		    
		    data.append((feat, label))
		return data


	def print_svm_meta_data(self, svm):
		print("test accuracy ", svm.score(test_feat, test_label))
		y_pred = svm.predict(test_feat)
		print(classification_report(test_label, y_pred))
