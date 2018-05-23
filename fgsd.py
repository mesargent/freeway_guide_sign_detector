import yaml
import sys

from remove_red import RemoveRed
from detector import Detector
from HOGClassifier import HOGSVMClassifier
from regions_preprocessor import RegionsPreprocessor
from region_finder import RegionFinder

from sklearn import svm
import numpy as np
import cv2
import matplotlib.pyplot as plt

#TODO implement detect signs in images

def main():
	with open("config.yml", "r") as yaml_file:
		config = yaml.load(yaml_file)

	def detect_signs_in_images():
		#get preprocessors
		remove_red_processor = RemoveRed(config)
		regions_preprocessor = RegionsPreprocessor(config)
		region_finder = RegionFinder(config, [regions_preprocessor])		
		svm_classifier = HOGSVMClassifier(config)

		print("detecting signs in images folder {}/".format(config['image_info']['image_paths']['detect']))
		image = cv2.imread(config['image_info']['image_paths']['detect'] + "/191.jpg", 1)

		detector = Detector(config, [], region_finder,  svm_classifier)
		signs = detector.detect(image)
		detector.display_signs(signs, image)


	def detect_signs_in_video():
		print("detecting signs in video")

	dispatch = {'detect_images': detect_signs_in_images, 'detect_video': detect_signs_in_video}
	
	command = sys.argv[1]
	dispatch[command]()


if __name__ == '__main__':
	main()
