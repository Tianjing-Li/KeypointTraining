from __future__ import division
import argparse
import cv2
import numpy as np
import json
import copy
from numpy.core.umath import deg2rad
import math
import random
import os

# usage: python multiplication.py -j jsonfile.json -n 100 -o results -s save

DEBUGGING = True
SAMPLE_SIZE = 608	# n * 32


ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-j", "--json", required=True, help="Json file name")
ap.add_argument("-n", "--number", required=True, help="number of generated images")
ap.add_argument("-o", "--output", required=True, help="path to output and json file name")
ap.add_argument("-s", "--save", required=True, help="path to output .txt and json files")

args = vars(ap.parse_args())

result_dir = (args["output"])
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
filename_cnt = 0
result_json = []

txt_dir = args["save"]
if not os.path.exists(txt_dir):
	os.makedirs(txt_dir)

def enclosureRect(pts):
	inds_array = np.moveaxis(np.array(pts), -1, 0)
	xlist = inds_array[0]
	ylist = inds_array[1]
	x0 = min(xlist)
	y0 = min(ylist)
	x1 = max(xlist)
	y1 = max(ylist)
	return (x0, y0, x1, y1)

# return M
# ref: http://stackoverflow.com/questions/19470955/warping-an-image-using-roll-pitch-and-yaw/21909889
def warpMatrix( (srcwid,srchei), theta, phi, gamma, scale, fovy):
	st=math.sin(deg2rad(theta))
	ct=math.cos(deg2rad(theta))
	sp=math.sin(deg2rad(phi))
	cp=math.cos(deg2rad(phi))
	sg=math.sin(deg2rad(gamma))
	cg=math.cos(deg2rad(gamma))

	halfFovy=fovy*0.5;
	d=math.hypot(srcwid, srchei);
	sideLength=scale*d/math.cos(deg2rad(halfFovy));
	#	print 'sideLength', sideLength
	h=d/(2.0*math.sin(deg2rad(halfFovy)));
	n=h-(d/2.0);
	f=h+(d/2.0);
	
	#F = np.eye(4, dtype = np.float32)			# transformation matrix F
	Rtheta = np.eye(4, dtype = np.float32)	# around Z-axis by theta degrees
	Rphi = np.eye(4, dtype = np.float32)	# rotation matrix around X-axis by phi degrees
	Rgamma = np.eye(4, dtype = np.float32)			# rotation matrix around Y-axis by gamma degrees
	
	T = np.eye(4, dtype = np.float32)		# translation matrix along Z-axis by -h units
	P = np.zeros(shape = [4,4], dtype = np.float32)		# projection matrix
	
	Rtheta[0,0] = Rtheta[1,1] = ct
	Rtheta[0,1] = -st;Rtheta[1,0] = st

	Rphi[1,1] = Rphi[2,2] = cp
	Rphi[1,2] = -sp
	Rphi[2,1] = sp
	
	Rgamma[0,0] = Rgamma[2,2] = cg
	Rgamma[0,2] = sg
	Rgamma[2,0] = sg

	T[2,3] = -h

	P[0,0] = P[1,1] = 1.0/math.tan(deg2rad(halfFovy))
	P[2,2] = -(f+n)/(f-n)
	P[2,3] = -(2.0*f*n)/(f-n)
	P[3,2] = -1.0
	
	# Compose transformations
#	print 'P', P
#	print 'T',T
#	print 'Rphi', Rphi
#	print 'Rtheta', Rtheta
#	print 'Rgamma', Rgamma


	F= np.dot(np.dot(np.dot(np.dot(P, T), Rphi), Rtheta), Rgamma)							# Matrix-multiply to produce master matrix
	#	print 'F', F

	# Transform 4x4 points
	ptsIn = np.zeros(shape = [4*3], dtype = np.float32)
	halfW=srcwid/2
	halfH=srchei/2

	
	ptsIn[0]=-halfW;ptsIn[ 1]= halfH
	ptsIn[3]= halfW;ptsIn[ 4]= halfH
	ptsIn[6]= halfW;ptsIn[ 7]=-halfH
	ptsIn[9]=-halfW;ptsIn[10]=-halfH
	ptsIn[2]=ptsIn[5]=ptsIn[8]=ptsIn[11]=0			# Set Z component to zero for all 4 components

#	print 'ptsIn', ptsIn

	ptsInMat = np.array(ptsIn, dtype=np.float32).reshape(4,3)
	ptsInMat = np.array([ptsInMat])
	
	#	print 'ptsInMat', ptsInMat


	
	ptsOutMat = cv2.perspectiveTransform(ptsInMat,F)
	#	print 'ptsOutMat', ptsOutMat
	
	ptsOut = ptsOutMat.reshape(12)
	
	ptsInPt2f = np.zeros(shape = [4,2], dtype = np.float32)
	ptsOutPt2f = np.zeros(shape = [4,2], dtype = np.float32)

	for i in np.arange(4):
		ptIn = [ptsIn [i*3+0], ptsIn [i*3+1]]
		ptOut = [ptsOut[i*3+0], ptsOut[i*3+1]]

		ptsInPt2f[i,0] =  ptIn[0]+halfW
		ptsInPt2f[i,1] =  ptIn[1]+halfH

		ptsOutPt2f[i,0] = (ptOut[0]+1.0) * sideLength*0.5;
		ptsOutPt2f[i,1] = (ptOut[1]+1.0) * sideLength*0.5;
	
#	print 'ptsInPt2f', ptsInPt2f
#	print 'ptsOutPt2f', ptsOutPt2f

	M = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

	corners = np.array(ptsOutPt2f)

	return M, corners

def warpImageWithCanvas(srcImg, theta, phi, gamma, scale, fovy, elements):
#						 vector<Point2f> &corners,
#						 vector<Point2f> cornersCanvas0,
#						 vector<Point2f> &cornersCanvas){
	halfFovy=fovy*0.5;
	d=math.hypot(np.size(srcImg, 1), np.size(srcImg, 0))
	sideLength=int(scale*d/math.cos(deg2rad(halfFovy))) 
	
	M, corners = warpMatrix(srcImg.shape[1::-1],theta,phi,gamma, scale,fovy)		#Compute warp matrix
	warpedImg = cv2.warpPerspective(srcImg, M, (sideLength,sideLength))    # WARP_INVERSE_MAP, BORDER_TRANSPARENT*/);//Do actual image warp

	# crop the warped image by percent
	percent = 0.80
	xoff = int(sideLength * (1 - percent) / 2.0)
	w = int(sideLength * percent)
	croppedImg = warpedImg[xoff:-xoff,xoff:-xoff]
	
	for story in elements[:]:
		canvas = story["contours"]
		canvas = np.array([canvas])

		#	print 'cornersCanvas0', cornersCanvas0
	
		canvas = cv2.perspectiveTransform(canvas, M)[0]

		story["contours"] = np.array([[x-xoff, y-xoff] for [x,y] in canvas])

	
	return croppedImg, elements





#https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
#def euler2mat(theta, phi, gamma)
def euler2mat(z=0, y=0, x=0):
	z = deg2rad(z)
	y = deg2rad(y)
	x = deg2rad(x)
	''' Return matrix for rotations around z, y and x axes
		
		Uses the z, then y, then x convention above
		
		Parameters
		----------
		z : scalar
		Rotation angle in radians around z-axis (performed first)
		y : scalar
		Rotation angle in radians around y-axis
		x : scalar
		Rotation angle in radians around x-axis (performed last)
		
		Returns
		-------
		M : array shape (3,3)
		Rotation matrix giving same rotation as for given angles
		
		Examples
		--------
		>>> zrot = 1.3 # radians
		>>> yrot = -0.1
		>>> xrot = 0.2
		>>> M = euler2mat(zrot, yrot, xrot)
		>>> M.shape == (3, 3)
		True
		
		The output rotation matrix is equal to the composition of the
		individual rotations
		
		>>> M1 = euler2mat(zrot)
		>>> M2 = euler2mat(0, yrot)
		>>> M3 = euler2mat(0, 0, xrot)
		>>> composed_M = np.dot(M3, np.dot(M2, M1))
		>>> np.allclose(M, composed_M)
		True
		
		You can specify rotations by named arguments
		
		>>> np.all(M3 == euler2mat(x=xrot))
		True
		
		When applying M to a vector, the vector should column vector to the
		right of M.  If the right hand side is a 2D array rather than a
		vector, then each column of the 2D array represents a vector.
		
		>>> vec = np.array([1, 0, 0]).reshape((3,1))
		>>> v2 = np.dot(M, vec)
		>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
		>>> vecs2 = np.dot(M, vecs)
		
		Rotations are counter-clockwise.
		
		>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
		>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
		True
		>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
		>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
		True
		>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
		>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
		True
		
		Notes
		-----
		The direction of rotation is given by the right-hand rule (orient
		the thumb of the right hand along the axis around which the rotation
		occurs, with the end of the thumb at the positive end of the axis;
		curl your fingers; the direction your fingers curl is the direction
		of rotation).  Therefore, the rotations are counterclockwise if
		looking along the axis of rotation from positive to negative.
		'''
	Ms = []
	if z:
		cosz = math.cos(z)
		sinz = math.sin(z)
		Ms.append(np.array([[cosz, -sinz, 0],
								[sinz, cosz, 0],
								[0, 0, 1]]))
	if y:
		cosy = math.cos(y)
		siny = math.sin(y)
		Ms.append(np.array([[cosy, 0, siny],
								[0, 1, 0],
								[-siny, 0, cosy]]))
	if x:
		cosx = math.cos(x)
		sinx = math.sin(x)
		Ms.append(np.array(	[[1, 0, 0],
						[0, cosx, -sinx],
						[0, sinx, cosx]]))
	if Ms:
		return reduce(np.dot, Ms[::-1])
	return np.eye(3)


def mat2euler(M, cy_thresh=None):
	''' Discover Euler angle vector from 3x3 matrix
		
		Uses the conventions above.
		
		Parameters
		----------
		M : array-like, shape (3,3)
		cy_thresh : None or scalar, optional
		threshold below which to give up on straightforward arctan for
		estimating x rotation.  If None (default), estimate from
		precision of input.
		
		Returns
		-------
		z : scalar
		y : scalar
		x : scalar
		Rotations in radians around z, y, x axes, respectively
		
		Notes
		-----
		If there was no numerical error, the routine could be derived using
		Sympy expression for z then y then x rotation matrix, which is::
		
		[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
		[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
		[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
		
		with the obvious derivations for z, y, and x
		
		z = atan2(-r12, r11)
		y = asin(r13)
		x = atan2(-r23, r33)
		
		Problems arise when cos(y) is close to zero, because both of::
		
		z = atan2(cos(y)*sin(z), cos(y)*cos(z))
		x = atan2(cos(y)*sin(x), cos(x)*cos(y))
		
		will be close to atan2(0, 0), and highly unstable.
		
		The ``cy`` fix for numerical instability below is from: *Graphics
		Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
		0123361559.  Specifically it comes from EulerAngles.c by Ken
		Shoemake, and deals with the case where cos(y) is close to zero:
		
		See: http://www.graphicsgems.org/
		
		The code appears to be licensed (from the website) as "can be used
		without restrictions".
		'''
	M = np.asarray(M)
	if cy_thresh is None:
		try:
			cy_thresh = np.finfo(M.dtype).eps * 4
		except ValueError:
			cy_thresh = _FLOAT_EPS_4
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	# cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	cy = math.sqrt(r33*r33 + r23*r23)
	if cy > cy_thresh: # cos(y) not close to zero, standard form
		z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
		y = math.atan2(r13,  cy) # atan2(sin(y), cy)
		x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
	else: # cos(y) (close to) zero, so x -> 0.0 (see above)
		# so r21 -> sin(z), r22 -> cos(z) and
		z = math.atan2(r21,  r22)
		y = math.atan2(r13,  cy) # atan2(sin(y), cy)
		x = 0.0
	return z, y, x

# DRAWS BLUE BOX COORDINATES BORDER
def drawPolygonOnImage(image, polygon, color=(0, 0, 255)):
	polygonlist = polygon.astype(np.int).tolist()
	
	#	print 'polygonlist', polygonlist
	
	polytuple = [tuple(i) for i in polygonlist];
	#	print 'polytuple', polytuple
	for i in np.arange(1, len(polytuple) +1):
		cv2.line(image, polytuple[i-1], polytuple[i%4], color, 1)

	return image

# transform method to modify an image
# TAKES IN MULTIPLE CANVAS POSITIONS NOW
def transform(srcImg, elements, num):
	global result_dir, filename_cnt, result_json

	
	# iterate the number of transforms we need
	for i in np.arange(num):
		# random transforms
		if i == 0:
			theta = -0; phi = 0; gamma = 0
			fovy = 0.1
		else:
			theta = random.randrange(-10, 10, 1); phi = random.randrange(-10, 10, 1); gamma = random.randrange(-10, 10, 1)	# random
			fovy = 0.1

		el = copy.deepcopy(elements)

		warpedImg, warpedElements = warpImageWithCanvas(srcImg, theta, phi, gamma, 1, fovy, el)

		h, w = warpedImg.shape[:2]
		k = SAMPLE_SIZE / w
		warpedImg = cv2.resize(warpedImg, (SAMPLE_SIZE, SAMPLE_SIZE))

		item_dict = {}
		item_dict["elements"] = []

		for story in warpedElements:
			canvas = story["contours"]
			canvas = np.array([[k * x, k * y] for [x, y] in canvas.tolist()])

			if DEBUGGING:	
				# Draw warped canvas position
				warpedImg = drawPolygonOnImage(warpedImg, canvas, (0,255,255))

			rect_dict = {}
			rect_dict["label"] = story["label"]

			(x0, y0, x1, y1) = enclosureRect(canvas)

			rect_dict["x1"] = round(x0)
			rect_dict["y1"] = round(y0)
			rect_dict["x2"] = round(x1)
			rect_dict["y2"] = round(y1)
			
			item_dict["elements"].append(rect_dict)

			# box coordinates of canvas
			box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
		
			# Draws GREEN BOX COORDINATES
			if DEBUGGING:
				warpedImg = drawPolygonOnImage(warpedImg, box, (0,255,0))

	
		cv2.imshow("warped", warpedImg)
		

		img_end = "/image_" + str(filename_cnt) + ".jpg"
		txt_end = "/image_" + str(filename_cnt) + ".txt"

		res_img_path = result_dir + img_end
		item_dict["image_path"] = res_img_path
		
		# save JSON
		result_json.append(item_dict)

		# save image
		cv2.imwrite(res_img_path, warpedImg)
		filename_cnt = filename_cnt + 1

		# save .txt
		txt = open(txt_dir + txt_end, "w")

		for rectangle in item_dict["elements"]:
			label = rectangle["label"]
			x1 = rectangle["x1"]
			x2 = rectangle["x2"]
			y1 = rectangle["y1"]
			y2 = rectangle["y2"]

			x = ((x1 + x2) / 2) / SAMPLE_SIZE
			y = ((y1 + y2) / 2) / SAMPLE_SIZE
			w = (x2 - x1) / SAMPLE_SIZE
			h = (y2 - y1) / SAMPLE_SIZE

			line = "{0} {1} {2} {3} {4}".format(label, x, y, w, h)

			txt.write(line)

			txt.write("\n")

		txt.close()

# open JSON file and get number of images
with open(args["json"]) as data_file:
	jsondata = json.load(data_file)
	num_items = len(jsondata)
	print 'Number of input images:', num_items

# iterate through all items in JSON

n = int(int(args["number"]) / num_items)

for jsonitem in jsondata:
	# print 'image', jsonitem["image_path"]

	image = cv2.imread(jsonitem["image_path"])

	#r efPt = jsonitem["contour"]
	elements = jsonitem["elements"]
	
	# for every possible element on an image
	for story in elements:
		refPt = story["contours"]
		refPt = np.array(refPt)

		# DRAWS ORIGINAL CONTOUR IN RED
		if DEBUGGING:
			image = drawPolygonOnImage(image, refPt)

		# transform into floating point
		story["contours"] = refPt.astype(np.float32).copy()

	# transform must take in a list of elements now
	transform(image, elements[:], n)

#M, corners = warpMatrix((10,10), 0, 0, 0, 1, 60000000000)
#M = np.around(M * 100) / 100.0
#
#print 'M', M
#print 'corners', corners

with open(txt_dir + "/" + result_dir + ".json", "w") as outfile:
	json.dump(result_json, outfile)

print "Number of output images: " + str(filename_cnt)


