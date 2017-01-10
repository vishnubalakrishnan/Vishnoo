import argparse
import cv2
import pickle
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


query = cv2.imread(args["query"])
query_gray = cv2.cvtColor(query,cv2.COLOR_BGR2GRAY)
query_blurred = cv2.GaussianBlur(query_gray, (3, 3), 0)
query_auto = auto_canny(query_blurred)

results = {}
i=0
x1 = pickle.load(open(args["index"],"r"))
for img in x1:
	val = cv2.matchShapes(query_auto, img, 3, 0.0)
	results[i] = val
	i=i+1

results = sorted([(v, k) for (k, v) in results.items()])

cv2.imshow("Query", query)

for (score, resultID) in results[:10]:
	# load the result image and display it
	print score
	# result = cv2.imread(args["result_path"] + "/" + resultID)
	# cv2.imshow("Result", result)
	# cv2.waitKey(0)
