# USAGE
# python search.py --index index.csv --query queries/103100.png --result-path dataset --core cores

# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher_parallel import Searcher
import argparse
import cv2
import pickle
import os.path
import re
import numpy as np
from PIL import Image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-e", "--edge", required = True,
	help = "Path to where the computed edge will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
ap.add_argument("-c", "--cores", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

# load the query image and describe it
query = cv2.imread(args["query"])

q = args["query"]
imageID = re.search('/(.+?).jpg', q).group(1)

if(os.path.isfile("out.bmp") == False):
	rf = Image.new( 'RGB', (1001,1000), "black")
	rf.save('out.bmp')


rf = Image.open("out.bmp")
pixels = rf.load() # create the pixel map
width, height = rf.size

bm = pixels[1000,int(imageID)]

#Check if query is already run once
if(bm[0] == 255 and bm[1] == 255 and bm[2]==255):
	cv2.imshow("Query", query)
	for y in range(width):
		rm = pixels[y,int(imageID)]
		if(rm[0] == 255 and rm[1] == 255 and rm[2]==255 and y!=1000):
			result = cv2.imread(args["result_path"] + "/" + str(y) + ".jpg")
			cv2.imshow("Result", result)
			k = cv2.waitKey(0)		# 1113938 -> +   1113940 -> -
			if(k-1113938 == 0):
				pixels[y,int(imageID)] = (255,255,255)
			else:
				if(k == 1113940):
					pixels[int(res),int(imageID)] = (100,100,100)

#If query run for the first time
else:
	query_gray = cv2.cvtColor(query,cv2.COLOR_BGR2GRAY)
	query_blurred = cv2.GaussianBlur(query_gray, (3, 3), 0)
	query_auto = auto_canny(query_blurred)
	features = cd.describe(query)

# perform the search
	searcher = Searcher(args["index"])
	results = searcher.search(features,query_auto,args["edge"],args["cores"])

# result_edge = {}
# i=0
# x1 = pickle.load(open(args["edge"],"r"))
# for img in x1:
# 	val = cv2.matchShapes(query_auto, img, 3, 0.0)
# 	result_edge[i] = val
# 	i=i+1
# result_edge = sorted([(v, k) for (k, v) in result_edge.items()])


# display the query
	cv2.imshow("Query", query)
# for (score, resultID) in result_edge[:10]:
# 	# load the result image and display it
# 	print score
# print imageID

# rf = Image.new( 'RGB', (1000,1000), "black") # create a new black image
# bm = pixels[1000,int(imageID)]
# print bm[0]
# print bm[1]
	pixels[1000,int(imageID)] = (255,255,255)
	y=20
# loop over the results
	# for (score, resultID) in results[:y]:
	# # load the result image and display it
	# 	result = cv2.imread(args["result_path"] + "/" + resultID)
	# 	res = re.search('(.+?).jpg', resultID).group(1)
	# 	cv2.imshow("Result", result)
	# 	k = cv2.waitKey(0)		# 1113938 -> +   1113940 -> -
	# 	if(k-1113938 == 0):
	# 		pixels[int(res),int(imageID)] = (255,255,255)

	ch = 'y'
	count=0
	while(ch == 'y'):
		y = y+count
		count = 0
		for (score, resultID) in results[:y]:
	# load the result image and display it
			result = cv2.imread(args["result_path"] + "/" + resultID)
			res = re.search('(.+?).jpg', resultID).group(1)
			rm = pixels[int(res),int(imageID)]
			if(rm[0] != 100 and rm[1] != 100 and rm[2]!=100):
				cv2.imshow("Result", result)
				k = cv2.waitKey(0)		# 1113938 -> +   1113940 -> -
				if(k-1113938 == 0):
					pixels[int(res),int(imageID)] = (255,255,255)
				else: 
					if(k == 1113940):
						pixels[int(res),int(imageID)] = (100,100,100)
						count = count+1
		ch = raw_input('Retry? ')


# im = np.array(rf)
rf.save('out.bmp')

# cv2.imshow("RF",im)
# cv2.waitKey(0)
