import cv2
import numpy as np 
import os
import shutil
import multiscale_detect as md
import json
import itertools
from random import randint


#returns a dictionary of imagepaths -> array of bounding boxes 
def getBoundingBoxesForImages(jsonString):
	boxes = {}
	jsonData = json.loads(jsonString)
	result = {x["filename"]: annotationsToTuples(x["annotations"]) for x in jsonData}
	return result

#convenience method to convert an array of annotation dicts found  in the json file
#to tuples of coordinates for a bounding rectangle, (x1, y1, x2, y2)
def annotationsToTuples(annotationArray):
	result = [(a['x'], a['y'], a["width"], a["height"]) for a in annotationArray]
	return result

#gets a list of sign images from scenes containing signs and json file of labels
def getImagesFromJSON(jsonString, imgdir=os.getcwd()):
	signs = []
	boxes = getBoundingBoxesForImages(jsonString)
	for key, arr in boxes.iteritems():
		scene = cv2.imread(imgdir + "/" + key)
		if scene is None:
			continue
		for coord in arr:
			x, y, width, height = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3]) 
			if width < 40 or height < 40: 
				continue			
			img = scene[y:y+height, x: x+width, :]
			signs.append(img)
	return signs


def getHeightsWidths(jsonString):
	jsonData = json.loads(jsonString)
	result =  [[(a["width"], a["height"]) for a in elem["annotations"]] for elem in jsonData]
	result = [item for sublist in result for item in sublist]
	return result


def getAllFiles(sourceDirectory):
	files = filter( lambda f: not f.startswith('.'), os.listdir(sourceDirectory))
	return files

def getMultiple(directory, num, label):
	data = []
	for i in range(num):
		filename, img = getRandomImage(directory)
		#tuple with filename, image, and label
		data.append((directory + "/" + filename, img, label))	
		cv2.imwrite(directory + "/" + filename, img)
	return data



if __name__ == "__main__":
	images = getImagesFromJSON(open("labels.json").read())
	print images[0].shape

	



	# #set parameters for hog
	# winSize = (80,80)
	# blockSize = (16,16)
	# blockStride = (8,8)
	# cellSize = (8,8)
	# nbins = 9
	# derivAperture = 1
	# winSigma = -1
	# histogramNormType = 0
	# L2HysThreshold = 2.0000000000000001e-01
	# gammaCorrection = 0
	# nlevels = 64
	# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	#                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)




	
	