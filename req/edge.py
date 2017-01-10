# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import pickle
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())
 
# loop over the images
# for imagePath in glob.glob(args["images"] + "/*.jpg"):
# 	# load the image, convert it to grayscale, and blur it slightly
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
# 	# apply Canny edge detection using a wide threshold, tight
# 	# threshold, and automatically determined threshold
# 	wide = cv2.Canny(blurred, 10, 200)
# 	tight = cv2.Canny(blurred, 225, 250)
# 	auto = auto_canny(blurred)
 
# 	# show the images
# 	cv2.imshow("Original", image)
# 	cv2.imshow("Edges", np.hstack([wide, tight, auto]))
# 	cv2.waitKey(0)
#loop over images
temp=[]
count=0
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	# describe the image
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	auto = auto_canny(blurred)
	temp.append(auto)
	count = count + 1
### For PNG files
# for imagePath in glob.glob(args["dataset"] + "/*.png"):
# 	imageID = imagePath[imagePath.rfind("/") + 1:]
# 	image = cv2.imread(imagePath)

# 	# describe the image
# 	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# 	auto = auto_canny(blurred)
# 	# temp.append(imageID)
# 	# temp.append(",")
# 	temp.append(auto)

pickle.dump(temp,open(args["index"],"w"))

print("Canny Edges for dataset stored")
print(count)

###################################################################################
# img1 = cv2.imread('38.jpg')
# img2 = cv2.imread('42.jpg')
# gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# blurred1 = cv2.GaussianBlur(gray1, (3, 3), 0)
# blurred2 = cv2.GaussianBlur(gray2, (3, 3), 0)
# auto1 = auto_canny(blurred1)
# # pickle.dump(auto1,open("contours.pkl","w"))
# temp=[]
# temp.append(auto1)
# auto2 = auto_canny(blurred2)
# temp.append(auto2)
# pickle.dump(temp,open(args["index"],"w"))

# x1 = pickle.load(open(args["index"],"r"))
# auto3 = x1[0]
# auto4 = x1[1]

# val=cv2.matchShapes(auto1, auto2, 3, 0.0)
# print val

# val1=cv2.matchShapes(auto3, auto4, 3, 0.0)
# print val1

# cv2.imshow("Edges", np.hstack([auto1, auto2, auto3, auto4]))
# # cv2.imshow("Edges", np.hstack([auto3, auto4]))
# cv2.waitKey(0)
##################################################################################