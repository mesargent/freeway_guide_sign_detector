import yaml
import sys

#TODO implement detect signs in images

def main():
	with open("config.yml", "r") as yaml_file:
		config = yaml.load(yaml_file)

	def detect_signs_in_images():
		#get and run the detector
		print("detecting signs in images folder {}/".format(config['image_info']['image_paths']['detect']))

	def detect_signs_in_video():
		print("detecting signs in video")

	dispatch = {'detect_images': detect_signs_in_images, 'detect_video': detect_signs_in_video}
	
	command = sys.argv[1]
	dispatch[command]()


if __name__ == '__main__':
	main()