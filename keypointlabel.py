# USAGE
# python keypointlabel.py -i imagefolder/ -j jsonfile.json
# 'S' - 'Save' key to save current box to elements list
# 'N' - 'Next' key to next image (appends current image info to JSON) 
# saving a JSON object with image_path will override any previous JSON object with same image_path
# 'P' - 'Previous' key to backtrack image
# 'R' - 'Reset' key to reset the marking of current images
# 'Q' - 'Quit' key to quit

import argparse
import cv2
import numpy as np
import json 
import glob
import os

refPt = []
keypoints = []
JSON = []

def reset():
	global refPt, keypoints

	del refPt[:]
	del keypoints[:]

def orderpt():
	global refPt

	if refPt[0][0] > refPt[1][0]:
		refPt = refPt[::-1]

def addkp():
	global refPt, keypoints

	# assume p1 is top left, p2 is bottom right

	if len(refPt) == 2:
		p1 = refPt[0]
		p2 = refPt[1]

		x = (p1[0] + p2[0]) / 2
		y = (p1[1] + p2[1]) / 2
		w = abs(p2[0] - p1[0])
		h = abs(p2[1] - p1[1])

		keypoints.append([x, y, w, h])

def clicker(event, x, y, flags, param):
	global refPt, keypoints
	global image

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))

		addkp() 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("Image", image)

		del refPt[:]

def append_json(imagepath):
	global JSON, keypoints

	for i in xrange(len(JSON)):
		if JSON[i]["image_path"] == imagepath:
			JSON.pop(i)
			break

	json = {}
	json["image_path"] = imagepath
	json["keypoints"] = list(keypoints)

	JSON.append(json)

def slideshow(images):
	global refPt, keypoints, image, clean, JSON, jsonname

	index = 0

	image = cv2.imread(images[index])
	clean = image.copy()

	cv2.namedWindow("Image")
	cv2.setMouseCallback("Image", clicker)

	# loops until "Q" pressed
	while True:

		# display image and wait for key
		cv2.imshow("Image", image)

		key = cv2.waitKey(1) & 0xFF

		# go to next image
		if key == ord("n") or key == ord("N"):
			if len(keypoints) > 0:
				append_json(images[index]) # saves current 
			else:
				print "No keypoints"
			if index < len(images) - 1:
				index = index + 1
				image = cv2.imread(images[index])

				# prepare frame for new image
				clean = image.copy()
				reset()
				# RESET CLICKER FOR NEW IMAGE
			else:
				os.system("say 'No more pictures'")

		# go to previous image
		elif key == ord("p") or key == ord("P"):
			if index > 0:
				index = index - 1
				image = cv2.imread(images[index])

				# prepare frame for new image
				clean = image.copy()
				reset()
				# RESET CLICKER FOR NEW IMAGE
			else:
				os.system("say 'No previous pictures")

		elif key == ord("s") or key == ord("S"):
			if len(JSON) > 0:
				with open (jsonname, "w") as js:
					json.dump(JSON, js, indent=2)
					print "JSON saved"
		# reset keypoints
		elif key == ord("r") or key == ord("R"):
			reset()
			image = clean.copy()

		# quit
		elif key == ord("q") or key == ord("Q"):
			if len(JSON) > 0:
				with open (jsonname, "w") as js:
					json.dump(JSON, js, indent=2)
					print "JSON saved"
			break

def edit(jsonname):
	global refPt, keypoints, image, clean, JSON

	with open(jsonname) as js:
		JSON = json.load(js)

	index = 0

	path = JSON[index]["image_path"]
	keypoints = JSON[index]["keypoints"]

	image = cv2.imread(path)
	clean = image.copy()

	cv2.namedWindow("Image")
	cv2.setMouseCallback("Image", clicker)

	# loops until "Q" pressed
	while True:

		for kp in keypoints:
			cv2.rectangle(image, (kp[0] - kp[2]/2, kp[1] - kp[3]/2), (kp[0] + kp[2]/2, kp[1] + kp[3]/2), (0, 255, 0), 2)

		# display image and wait for key
		cv2.imshow("Image", image)

		key = cv2.waitKey(1) & 0xFF

		# go to next image
		if key == ord("n") or key == ord("N"):
			if len(keypoints) > 0:
				append_json(path) # saves current 
			else:
				print "No keypoints"
			if index < len(JSON) - 1:
				index = index + 1
				path = JSON[index]["image_path"]
				image = cv2.imread(path)
				keypoints = JSON[index]["keypoints"]
				# prepare frame for new image
				clean = image.copy()
				del refPt[:]
				# RESET CLICKER FOR NEW IMAGE
			else:
				os.system("say 'No more pictures'")

		# go to previous image
		elif key == ord("p") or key == ord("P"):
			if index > 0:
				index = index - 1
				path = JSON[index]["image_path"]
				image = cv2.imread(path)
				keypoints = JSON[index]["keypoints"]
				# prepare frame for new image
				clean = image.copy()
				del refPt[:]
				# RESET CLICKER FOR NEW IMAGE
			else:
				os.system("say 'No previous pictures")

		elif key == ord("s") or key == ord("S"):
			if len(keypoints) > 0:
				append_json(path)

				with open (jsonname, "w") as js:
					json.dump(JSON, js, indent=2)
					print "JSON saved"

		# reset keypoints
		elif key == ord("r") or key == ord("R"):
			reset()
			image = clean.copy()

		# quit
		elif key == ord("q") or key == ord("Q"):
			with open (jsonname, "w") as js:
					json.dump(JSON, js, indent=2)
					print "JSON saved"
			break
def main():
	global JSON, jsonname

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to image files")
	ap.add_argument("-j", "--json", required=True, help="Path to JSON file")
	ap.add_argument("-e", "--edit", default=False)

	args = vars(ap.parse_args())

	# write into output json
	jsonname = args["json"]
	if not jsonname.endswith(".json"):
		jsonname = jsonname + ".json"

	# list of images
	images = []
	for type in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
		for path in glob.glob(args["image"] + "/" + type):
			# JSON already exists
			if os.path.isfile(jsonname):
				with open(jsonname) as js:
					JSON = json.load(js)
			
			exists = False

			for i in xrange(len(JSON)):
				if JSON[i]["image_path"] == path:
					exists = True

			if not exists:	
				images.append(path)

	if not args["edit"]:
		slideshow(images)
	else:
		edit(args["json"])

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()