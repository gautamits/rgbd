import cv2
def histogram(image):
	#image = cv2.imread("test.jpg",0)
	winSize = (70,70)
	blockSize = (70,70)
	blockStride = (5,5)
	cellSize = (35,35)
	nbins = 64
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)
	#hist = hog.compute(image,winStride,padding,locations)
	hist=hog.compute(image)
	#print "histogram shape found by hog is ",hist.shape
	return hist