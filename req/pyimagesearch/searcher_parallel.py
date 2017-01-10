# import the necessary packages
import numpy as np
import csv
import pickle
import cv2
import itertools
from threading import Thread


def divide(iterable, parts):
		items = list(iterable)

		seqs = [[] for _ in xrange(parts)]
		while items:
		    for i in xrange(parts):
		        if not items:
		            break
		    
		        seqs[i].append(items.pop())
		    
		return seqs

class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	
	def parallel_search(self, reader, x1, queryFeatures, query, result):
		for row,img in itertools.izip(reader,x1):
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
				val = cv2.matchShapes(query, img, 3, 0.0)
				sum1 = d*0.6 + val*10

				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
				result[row[0]] = sum1


	def search(self, queryFeatures, query, edge, limit=20, cores=1):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			x1 = pickle.load(open(edge,"r"))

			reader_partition = divide(reader,cores)
			x1_partition = divide(x1,cores)

			threads = [None]*cores

			for i in xrange(cores) :
				threads[i] = Thread(target=self.parallel_search, args=(reader_partition[i], x1_partition[i],queryFeatures, query, results))
    			threads[i].start()

    		for i in xrange(cores) :
    			threads[i].join()

			# close the reader
			f.close()

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d